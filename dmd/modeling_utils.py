import pickle
import sys
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch.nn import Module
from torch.nn.functional import one_hot
from torch.optim import AdamW

from dmd import SOURCES_ROOT, dnnlib
from dmd.torch_utils import distributed as dist
from dmd.training.networks import EDMPrecond
from dmd.utils.common import seed_everything


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


def load_dmd_model(model_path: str, device: torch.device, for_training: bool = False, optimizer_kwargs: Dict[str, Any] = None) -> Union[Module, Tuple]:
    """
    Loads a pretrained DMD model from given path.

    Args:
        model_path (str): Path to the model weights.
        device (torch.device): Device to load the model to.
        for_training (bool): Whether to only load the distilled one-step diffusion model. Otherwise,
            returns a tuple of all model and optimizer required for training. [default: False]
        optimizer_kwargs (dict(str, any)): Optional keyword arguments to pass to the optimizer class (AdamW).
            This argument is ignored when `for_training` is set to `False`.
    """
    model_g = EDMPrecond(
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
    model_dict = torch.load(model_path, map_location="cpu")
    model_g.load_state_dict(model_dict["model_g"])
    model_g.to(device)
    if not for_training:
        return model_g.to(device)
    optimizer_kwargs = optimizer_kwargs or {}
    model_d = EDMPrecond(
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
    model_d.load_state_dict(model_dict["model_d"])
    model_d.to(device)
    optimizer_g = AdamW(model_g.parameters(), **optimizer_kwargs)
    optimizer_d = AdamW(model_d.parameters(), **optimizer_kwargs)
    optimizer_g.load_state_dict(model_dict["optimizer_g"])
    optimizer_d.load_state_dict(model_dict["optimizer_d"])
    return model_g, optimizer_g, model_d, optimizer_d


def encode_labels(class_ids: torch.Tensor, label_dim: int) -> Optional[torch.Tensor]:
    """One-hot encoding for given class ids."""
    class_labels = None
    if class_ids is None:
        return class_labels
    elif label_dim:
        class_labels = one_hot(class_ids, num_classes=label_dim)
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


def generate_samples(model, class_id: int, size: int = 25, seed: int = 42, device: str = "cpu"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.eval().to(device)
    seed_everything(seed)
    class_ids = torch.zeros(size, dtype=torch.int64, device=device)
    class_ids += class_id
    class_labels = encode_labels(class_ids, model.label_dim)
    z = torch.randn((size, 3, 32, 32), device=device)
    g_sigma = get_fixed_generator_sigma(size, device=device)
    z = z * g_sigma[0, 0]
    with torch.no_grad():
        out = model(z, g_sigma, class_labels=class_labels)
    return out
