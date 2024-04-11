# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import torch

from dmd.utils.array import torch_to_pillow


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """
    Constructs the noise schedule of Karras et al. (2022).
    Taken from openai/consistency_models
    https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py#L422
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


def edm_sampler(
    net,
    latents,
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
    class_labels=None,
    randn_like=torch.randn_like,
):
    """
    Proposed EDM sampler (Algorithm 2).
    """
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=latents.device)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


if __name__ == "__main__":
    from dmd.modeling_utils import load_model


    device = torch.device("cuda")
    latents = torch.randn(1,3,32,32, device=device)
    mu_real = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
                         device=device)
    samples = edm_sampler(
            mu_real,
            latents=latents,
    )
    torch_to_pillow(samples, 0).show()

