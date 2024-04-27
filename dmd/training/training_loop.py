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
import time
import copy
import json
import pickle
from contextlib import suppress
from typing import Optional

import psutil
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss as TorchLoss

from dmd import dnnlib
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.loss import GeneratorLoss, DenoisingLoss
from dmd.modeling_utils import load_model, get_sigmas_karras, forward_diffusion
from dmd.torch_utils import distributed as dist
from dmd.torch_utils import training_stats
from dmd.torch_utils import misc


def update_parameters(model, loss, optimizer, max_norm):
    if not math.isfinite(loss.item()):
        print("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)

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
    # neptune_run: Optional[Run] = None,
):
    amp_autocast = amp_autocast or suppress

    # Set G and mu_fake to train mode, mu_real should be frozen
    generator.train()
    mu_fake.train()
    mu_real.requires_grad_(False).eval()

    # metric_logger = utils.MetricLogger(delimiter="  ", neptune_run=neptune_run)
    # metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for pairs in data_loader:
        y_ref = pairs["image"].to(device, non_blocking=True)
        z_ref = pairs["latent"].to(device, non_blocking=True)
        z = torch.randn_like(y_ref, device=device)
        class_ids = pairs["class_id"].to(device, non_blocking=True)

        with amp_autocast():
            # Update generator
            # TODO: Generator is the same network but we cannot use EDM version
            #   as it depends on time (e.g. generator(z, sigma_t)). We need to
            #   preserve the weights but remove the time dependence.
            x = generator(z)
            x_ref = generator(z_ref)
            l_g = loss_g(mu_real, mu_fake, x, x_ref, y_ref)

        update_parameters(generator, loss_g, optimizer_g, max_norm)
        torch.cuda.synchronize()

        with amp_autocast():
            # Update mu_fake
            t = torch.randint(0, 1000, [x.shape[0]])
            x_t, sigma = forward_diffusion(x.detach(), t)  # stop grad
            pred_fake_image = mu_fake(x_t, sigma)
            w_dl = 1 / sigma ** 2 + 1 / mu_fake.sigma_data ** 2  # SNR + 1 / sigma_data^2 (Loss weighting)
            l_d = loss_d(pred_fake_image, x.detach(), w_dl)

        update_parameters(mu_fake, loss_d, optimizer_d, max_norm)
        torch.cuda.synchronize()

        # if model_ema is not None:
        #     model_ema.update(model)

        # metric_logger.update(loss=loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return {}


if __name__ == "__main__":
    data_path = "/home/devrim/lab/gh/ms/dmd/data/distillation_dataset_h5/cifar.hdf5"
    network_path = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
    device = torch.device("cuda")
    dataset = CIFARPairs(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    mu_real = load_model(network_path=network_path, device=device)
    mu_fake = load_model(network_path=network_path, device=device)
    generator = load_model(network_path=network_path, device=device)

    generator_loss = GeneratorLoss()
    diffusion_loss = DenoisingLoss()

    generator_optimizer = AdamW(params=generator.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    train_one_epoch(generator, mu_fake, mu_real, dataloader, generator_loss, diffusion_loss, generator_optimizer,
                    diffuser_optimizer, device, 1)
