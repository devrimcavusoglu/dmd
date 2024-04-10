import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss, MSELoss

from dmd.edm.modeling_utils import load_model
from dmd.edm.sampler import edm_sampler


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


class DistributionMatchingLoss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse_loss = MSELoss()

    def forward(self, mu_real: Module, mu_fake: Module, x: torch.Tensor, min_dm_step: int, max_dm_step: int, bs: int) -> torch.Tensor:
        timestep = torch.randint(min_dm_step, max_dm_step, [bs])
        noise = torch.randn_like(x, device=x.device)

        sigma = get_sigmas_karras(1000, sigma_min=0.002, sigma_max=80, rho=7.0, device=x.device)
        ns = noise * sigma[timestep, None, None, None]  # broadcast for scalar product
        noisy_x = x + ns

        with torch.no_grad():
            pred_fake_image = edm_sampler(mu_fake, noisy_x)
            pred_real_image = edm_sampler(mu_real, noisy_x)

        weighting_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3], keepdim=True)
        grad = (pred_fake_image - pred_real_image) / weighting_factor
        diff = (x - grad).detach()  # stop-gradient
        return 0.5 * self.mse_loss(x, diff)


class DenoisingLoss(_Loss):
    def forward(self, pred_fake_image: torch.Tensor, x: torch.Tensor, weight) -> torch.Tensor:
        # TODO: Implement weighting strategy
        weight = 1
        return torch.mean(weight * (pred_fake_image - x) ** 2)


if __name__ == "__main__":
    loss = DistributionMatchingLoss()
    device = torch.device("cuda")
    x = torch.randn(16, 3, 32, 32, device=device)  # B,C,W,H
    mu_real = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", device=device)
    mu_fake = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
                         device=device)
    l = loss(mu_real, mu_fake, x, 10, 100, 16)
    print(l.item())
