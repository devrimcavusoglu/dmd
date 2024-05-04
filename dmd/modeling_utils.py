import pickle
import sys
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.functional import one_hot

from dmd import SOURCES_ROOT, dnnlib
from dmd.torch_utils import distributed as dist
from dmd.training.networks import EDMPrecond
from dmd.utils.common import image_grid


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


def load_edm(model_path: str, device: torch.device) -> Module:
    """
    Loads a pretrained model from given path. This function loads the model from the original
    pickle file of EDM models.

    Note:
        The saved binary files (pickle) contain serialized Python object where some of them
        require certain imports. This module is not identical to the structure of 'NVLabs/edm',
        see the related comment.

    Args:
        model_path (str): Path to the model weights.
        device (torch.device): Device to load the model to.
    """
    # Refactoring the package structure and import scheme (e.g. this module) breaks the loading of the
    # pickle file (as it also possesses the complete module structure at the save time). The following
    # line is a little trick to make the import structure the same to load the pickle without a failure.
    sys.path.insert(0, SOURCES_ROOT.as_posix())
    with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
        return pickle.load(f)["ema"].to(device)


def load_dmd_model(model_path: str, device: torch.device) -> Module:
    """
    Loads a pretrained model from given path. This function loads the model from the original
    pickle file of EDM models.

    Note:
        The saved binary files (pickle) contain serialized Python object where some of them
        require certain imports. This module is not identical to the structure of 'NVLabs/edm',
        see the related comment.

    Args:
        model_path (str): Path to the model weights.
        device (torch.device): Device to load the model to.
    """
    m = torch.load(model_path, map_location="cpu")
    return m["model_g"].to(device)


def encode_labels(class_ids: torch.Tensor, label_dim: int) -> Optional[torch.Tensor]:
    class_labels = None
    if label_dim:
        one_hot(class_ids, num_classes=label_dim)
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
    ns = noise * sigma[-(t + 1), None, None, None]  # broadcast for scalar product
    noisy_x = x + ns
    return noisy_x, sigma[-(t + 1)]


def get_fixed_generator_sigma(size: int, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Returns sigmas of size `size` with fixed sigmas for the generator. In the paper, it
    is fixed to T-1'th timestep for generator. In practice EDM models are fed sigma value at timestep t.
    """
    sigma = get_sigmas_karras(n=1000, sigma_min=0.002, sigma_max=80.0, device=device)[1]  # sigma_(T-1)
    return torch.tile(sigma, (1, size))


def sample_from_generator(
    generator: EDMPrecond,
    seeds: List[int] = None,
    latents: torch.Tensor = None,
    class_ids: torch.Tensor = None,
    device: Union[str, torch.device] = None,
    im_channels: int = 3,
    im_resolution: int = 32,
    scale_latents: bool = True,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not latents or not seeds:
        raise ValueError("Either `latent` or `seeds` must be provided.")

    if latents is None:
        scale_latents = True  # override
        rnd = StackedRandomGenerator(device, seeds)
        latents = rnd.randn(
            [len(seeds), im_channels, im_resolution, im_resolution],
            device=device,
        )

    g_sigmas = get_fixed_generator_sigma(len(seeds), device=device)
    if scale_latents:
        latents = latents * g_sigmas[0, 0]

    return generator(latents, g_sigmas, class_labels=class_ids)


if __name__ == "__main__":
    device = torch.device("cuda")
    model = load_dmd_model("/home/devrim/lab/gh/ms/dmd/outputs/toy_test/best_checkpoint.pt", device)
    seeds = list(range(10))
    class_ids = torch.tensor([0] * len(seeds), device=device)
    encode_labels(class_ids=class_ids, label_dim=10)
    r = sample_from_generator(model, seeds=seeds, class_ids=class_ids, device=device)
    image_grid(r, 2, 5)
