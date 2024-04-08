# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Devrim Cavusoglu & Ahmet Burak Yıldırım
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".

This module is taken and adapted from https://github.com/NVLabs/edm with certain
changes.
"""
import dataclasses
import os
import pickle
import re
import sys
from typing import List, Optional, Union

import PIL.Image
import torch
import tqdm

from dmd import EDM_PACKAGE_DIR
from dmd.edm import dnnlib
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


@dataclasses.dataclass
class GenerationConfig:
    steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7.0
    S_churn: float = 0
    S_min: float = 0
    S_max: float = float("inf")
    S_noise: float = 1.0


class EDMGenerator:
    _config = None

    def __init__(self, network_path: str, device: str = None, load_on_init: bool = True):
        self.network_path = network_path
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.model = None
        if load_on_init:
            self.load_model(network_path, self.device)
        self.set_config()

    @property
    def config(self) -> Optional[GenerationConfig]:
        return self._config

    @property
    def current_model_device(self) -> Optional[torch.device]:
        """
        This function assumes all model parameters are on the same device.
        """
        if self.model is None:
            return
        return next(self.model.parameters()).device

    def set_config(self, **kwargs):
        """
        Sets the generation config. For parameters see `GenerationConfig`.
        """
        current_params = self.config.__dict__ if self.config is not None else {}
        params = {**current_params, **kwargs}  # overwrite current config
        self._config = GenerationConfig(**params)

    def load_model(self, network_path: str, device: torch.device) -> None:
        """
        Loads a pretrained model from given path. This function loads the model from the original
        pickle file of EDM models.

        Note:
            The saved binary files (pickle) contain serialized Python object where some of them
            require certain imports. This module is not identical to the structure of 'NVLabs/edm',
            see the related comment.

        Args:
            network_path (str): Path to the model weights.
            device (torch.device): Device to load the model to.
        """
        if self.model is not None:
            raise RuntimeError(f"Model '{self.network_path}' is already loaded. To load a new model, "
                               f"use 'unload_model' first.")

        # Refactoring the package structure and import scheme (e.g. this module) breaks the loading of the
        # pickle file (as it also possesses the complete module structure at the save time). The following
        # line is a little trick to make the import structure the same to load the pickle without a failure.
        sys.path.insert(0, EDM_PACKAGE_DIR.as_posix())
        with dnnlib.util.open_url(network_path, verbose=(dist.get_rank() == 0)) as f:
            self.model = pickle.load(f)['ema'].to(device)

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    @staticmethod
    def _parse_int_list(s: Union[str, List[int]]):
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

    def __call__(self,
        outdir: str,
        subdirs: bool = False,
        seeds: Union[List[int], str] = "0-63",
        class_idx: Optional[int] = None,
        max_batch_size: int = 64,
        device: Optional[str] = None,
        save_format: Optional[str] = "images",
        **kwargs,
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
            outdir (str): Where to save the output images.
            seeds (Union(List(int), str): Random seeds (e.g. 1,2,5-10).
            subdirs (bool): If true, creates subdirectory for every 1000 seeds.
            class_idx (int): Class label  [default: random].
            max_batch_size (int): Maximum batch size. [default: 64]
            save_format (Optional(str)): Format to save to `outdir`. [default: 'images']
                - None: no saving, and generated samples are returned.
                - 'images': the generated images are saved.
                - 'all': the generated images and latents are saved together (for precomputing).
            **kwargs: Additional parameters for generation config, see `GenerationConfig`.

        Returns:
            pass
        """
        dist.init()
        seeds = self._parse_int_list(seeds)
        num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

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

            latents, images = self.generate_batch(seeds=batch_seeds, class_idx=class_idx, **kwargs)

            # Save images.
            if save_format == "images":
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = os.path.join(outdir, f"{seed-seed%1000:06d}") if subdirs else outdir
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"{seed:06d}.png")
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, "RGB").save(image_path)
            elif save_format == "all":
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = os.path.join(outdir, f"{seed - seed % 1000:06d}") if subdirs else outdir
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"{seed:06d}.png")
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, "RGB").save(image_path)

        # Done.
        torch.distributed.barrier()
        dist.print0("Done.")

    def generate_batch(self, seeds: List[int], class_idx: Optional[int] = None, **kwargs):
        self.set_config(**kwargs)
        device = self.current_model_device
        batch_size = len(seeds)

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn(
            [batch_size, self.model.img_channels, self.model.img_resolution, self.model.img_resolution],
            device=device,
        )
        class_labels = None
        if self.model.label_dim:
            class_labels = torch.eye(self.model.label_dim, device=device)[
                rnd.randint(self.model.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        images = edm_sampler(
            self.model,
            latents,
            steps=self.config.steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
            S_churn=self.config.S_churn,
            S_min=self.config.S_min,
            S_max=self.config.S_max,
            S_noise=self.config.S_noise,
            class_labels=class_labels,
            randn_like=rnd.randn_like,
        )
        return latents, images
