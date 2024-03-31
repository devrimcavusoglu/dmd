# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
from typing import List, Optional, Union

import PIL.Image
import torch
import tqdm

from dmd.edm.networks import EDMPrecond
from dmd.edm.sampler import edm_sampler
from dmd.edm.torch_utils import distributed as dist


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators]
        )


def parse_int_list(s):
    """
    Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def load_model(network_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a pretrained model from given path. This function is created as original
    EDM models are saved as pkl and as a whole, and thus refactoring the Python module
    breaks it. Instead, this function is loading the model from only saved weights, and
    will not break. However, the weights model dependant, so currently this function
    only supports EDM CIFAR conditioned model (VP). Other models may be added as desired.

    Args:
        network_path (str): Path to the model weights.
        device (torch.device): Device to load the model to.

    Returns:
        The loaded model.
    """
    network = EDMPrecond(
        img_resolution=32,
        img_channels=3,
        label_dim=10,
        resample_filter=[1, 1],
        embedding_type="positional",
        augment_dim=9,
        dropout=0.13,
        model_type="SongUNet",
        encoder_type="standard",
        channel_mult_noise=1,
        model_channels=128,
        channel_mult=(2, 2, 2),
    )
    network.load_state_dict(torch.load(network_path))
    return network.to(device)


def generate(
    network_path: str,
    outdir: str,
    subdirs: bool = False,
    seeds: Union[List[int], str] = "0-63",
    class_idx: Optional[int] = None,
    max_batch_size: int = 64,
    device: Optional[str] = None,
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7.0,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
):
    """
    Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    Args:
        network (str): Network pickle filename.
        outdir (str): Where to save the output images.
        seeds (Union(List(int), str): Random seeds (e.g. 1,2,5-10).
        subdirs (bool): If true, creates subdirectory for every 1000 seeds.
        class_idx (int): Class label  [default: random].
        max_batch_size (int): Maximum batch size. [default: 64]
        steps (int): Number of sampling steps. [default: 18]
        sigma_min (float): Lowest noise level. [default: varies]
        sigma_max (float): Highest noise level. [default: varies]
        rho (float): Time step exponent. [default: 7.0]
        S_churn (float): Stochasticity strength. [default: 0.0]
        S_min (float): Stochasticity min noise level. [default: 0.0]
        S_max (float): Stochasticity max noise level. [default: float('inf')]
        S_noise (float): Stochasticity noise inflation. [default: 1.0]

    Returns:
        pass
    """
    dist.init()
    seeds = parse_int_list(seeds)
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    if not isinstance(device, torch.device):
        raise ValueError("`device` must be an instance of `torch.device` or a string conforming `torch.device`.")

    # Load network.
    dist.print0(f'Loading network from "{network_path}"...')
    network = load_model(network_path_or_url=network_path, device=device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, network.img_channels, network.img_resolution, network.img_resolution], device=device
        )
        class_labels = None
        if network.label_dim:
            class_labels = torch.eye(network.label_dim, device=device)[
                rnd.randint(network.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        images = edm_sampler(
            network,
            latents,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            S_churn=S_churn,
            S_min=S_min,
            S_max=S_max,
            S_noise=S_noise,
            class_labels=class_labels,
            randn_like=rnd.randn_like,
        )

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f"{seed-seed%1000:06d}") if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed:06d}.png")
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
            else:
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0("Done.")
