import pickle
import sys
from typing import Tuple

import torch
from torch.nn import Module

from dmd import SOURCES_ROOT, dnnlib
from dmd.torch_utils import distributed as dist


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


def load_model(network_path: str, device: torch.device) -> Module:
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
    # Refactoring the package structure and import scheme (e.g. this module) breaks the loading of the
    # pickle file (as it also possesses the complete module structure at the save time). The following
    # line is a little trick to make the import structure the same to load the pickle without a failure.
    sys.path.insert(0, SOURCES_ROOT.as_posix())
    with dnnlib.util.open_url(network_path, verbose=(dist.get_rank() == 0)) as f:
        return pickle.load(f)["ema"].to(device)


def copy_weights(original: Module, clone: Module) -> None:
    """
    Copies weights from `original` to `clone`.
    """
    clone.load_state_dict(original.state_dict())


def encode_labels(class_ids: torch.Tensor, label_dim: int) -> torch.Tensor:
    batch_size = class_ids.shape[-1]
    class_labels = None
    if label_dim:
        class_labels = torch.zeros((batch_size, label_dim), device=class_ids.device)
        class_labels[:, class_ids] = 1
    return class_labels


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):

    """
    Constructs the noise schedule of Karras et al. (2022).
    Taken from crowsonkb/k-diffusion, An implementation of Karras et al. (2022) in PyTorch.
    https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L17
    """

    def append_zero(x):
        """
        Appends zero to the end of the input array.
        """
        return torch.cat([x, x.new_zeros([1])])

    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def forward_diffusion(
    x: torch.Tensor, t: torch.Tensor, n: int = 1000, noise: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x, device=x.device)

    sigma = get_sigmas_karras(n, sigma_min=0.002, sigma_max=80, rho=7.0, device=x.device)  # N, N-1, ..., 0
    ns = noise * sigma[-(t+1), None, None, None]  # broadcast for scalar product
    noisy_x = x + ns
    return noisy_x, sigma[-(t+1)]
