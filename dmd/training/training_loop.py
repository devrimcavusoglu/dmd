# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""
import math
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional

import PIL.Image
import torch
from neptune import Run
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss as TorchLoss

from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.loss import GeneratorLoss, DenoisingLoss
from dmd.modeling_utils import load_model, forward_diffusion, encode_labels
from dmd.utils.logging import MetricLogger


def _save_array_as_images(
        output_dir: str,
        images: torch.Tensor,
        prefix: str,
):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    images_output_dir = Path(output_dir)
    images_output_dir.mkdir(exist_ok=True, parents=True)
    for i, image_np in enumerate(images_np):
        image_path = images_output_dir / f"{prefix}_{i:06d}.png"
        if image_np.shape[2] == 1:
            PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
        else:
            PIL.Image.fromarray(image_np, "RGB").save(image_path)


def update_parameters(model, loss, optimizer, max_norm):
    optimizer.zero_grad()

    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    loss.backward(create_graph=is_second_order)
    if max_norm is not None:
        clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()


def train_one_epoch(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # loss_scaler,
    max_norm: float = 10,
    model_ema: Optional = None,
    amp_autocast=None,
    neptune_run: Optional[Run] = None,
):
    amp_autocast = amp_autocast or suppress

    # Set G and mu_fake to train mode, mu_real should be frozen
    generator.requires_grad_(True).train()
    mu_fake.requires_grad_(True).train()
    mu_real.requires_grad_(False).eval()

    metric_logger = MetricLogger(delimiter="  ", neptune_run=neptune_run)
    # metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    im_save_freq = 50

    i = 0
    for pairs in metric_logger.log_every(data_loader, print_freq, header):
        y_ref = pairs["image"].to(device, non_blocking=True).to(torch.float32).clip(-1, 1)
        z_ref = pairs["latent"].to(device, non_blocking=True).to(torch.float32)
        z = torch.randn_like(y_ref, device=device)
        # Scale Z ~ N(0,1) (z and z_ref) w/ 80.0 to match the sigma_t at T_n
        z = z * 80.0
        z_ref = z_ref * 80.0
        class_idx = pairs["class_id"].to(device, non_blocking=True)
        class_ids = encode_labels(class_idx, generator.label_dim)

        with amp_autocast():
            # Update generator

            # We kept the generator (G) the same as the pretrained EDM model, which depends on
            # sigma_t. We kept this sigma_t constant (80) in the training, and it should be used
            # as is in the inference/sampling as well. There is no information how G is constructed,
            # another alternative could've been to employ a equal size of UNet (SongUNet) and
            # copy weights without time dependence, but this approach seem to have gaps between
            # starting from the exact backbone vs. some blocks copied (e.g. no positional encoding).
            sigmas = torch.tensor([80.0] * z.shape[0], device=device)
            x = generator(z, sigmas, class_labels=class_ids)
            x_ref = generator(z_ref, sigmas, class_labels=class_ids)
            l_g = loss_g(mu_real, mu_fake, x, x_ref, y_ref)
            if not math.isfinite(l_g.item()):
                print(f"Generator Loss is {l_g.item()}, stopping training")
                sys.exit(1)

        update_parameters(generator, l_g, optimizer_g, max_norm)
        torch.cuda.synchronize()

        with amp_autocast():
            # Update mu_fake
            t = torch.randint(1, 1000, [x.shape[0]])  # t ~ DU(1,1000) as t=0 leads 1/0^2 -> inf
            x_t, sigma_t = forward_diffusion(x.detach(), t)  # stop grad
            l_d = loss_d(mu_fake, x.detach(), sigma_t)
            if not math.isfinite(l_d.item()):
                print(f"Diffusion Loss is {l_d.item()}, stopping training")
                sys.exit(1)

        update_parameters(mu_fake, l_d, optimizer_d, max_norm)
        torch.cuda.synchronize()

        if i % im_save_freq == 0:
            output_dir = Path("/home/devrim/lab/gh/ms/dmd/data/toy") / f"iter_{i}"
            _save_array_as_images(output_dir.as_posix(), x, "x")
            _save_array_as_images(output_dir.as_posix(), x_ref, "x_ref")
            with torch.no_grad():
                real_pred = mu_real(x_t, sigma_t, class_labels=class_ids)
                fake_pred = mu_fake(x_t, sigma_t, class_labels=class_ids)
            _save_array_as_images(output_dir.as_posix(), real_pred, "x_real")
            _save_array_as_images(output_dir.as_posix(), fake_pred, "x_fake")
            _save_array_as_images(output_dir.as_posix(), y_ref, "y_ref")

        # if model_ema is not None:
        #     model_ema.update(model)

        metric_logger.update(loss_g=l_g.item(), loss_d=l_d.item())
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        i += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    data_path = "/home/devrim/lab/gh/ms/dmd/data/distillation_dataset_h5/cifar.hdf5"
    network_path = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
    device = torch.device("cuda")
    dataset = CIFARPairs(data_path)
    # TODO: BS -> 64
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    mu_real = load_model(network_path=network_path, device=device)
    mu_fake = load_model(network_path=network_path, device=device)
    generator = load_model(network_path=network_path, device=device)

    generator_loss = GeneratorLoss()
    diffusion_loss = DenoisingLoss()

    generator_optimizer = AdamW(params=generator.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    train_one_epoch(generator, mu_fake, mu_real, dataloader, generator_loss, diffusion_loss, generator_optimizer,
                    diffuser_optimizer, device, 1)
