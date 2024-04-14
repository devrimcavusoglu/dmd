import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.loss import _Loss

from dmd.modeling_utils import forward_diffusion


class DistributionMatchingLoss(_Loss):
    """
    Loss function for DMD (Algorithm 2) proposed in
    "One-step Diffusion with Distribution Matching Distillation".

    Note:
        There is no explicit mention of how/what `min_dm_step` and `max_dm_step` are chosen.
        Here we conducted a loose range grid search over 0-1000 discrete steps, and observed the
        noisiness of input images, and then set an intuitive min and max steps following the
        observations.
    """

    def forward(
        self, mu_real: Module, mu_fake: Module, x: torch.Tensor, min_dm_step: int = 200, max_dm_step: int = 300
    ) -> torch.Tensor:
        bs = x.shape[0]
        timestep = torch.randint(min_dm_step, max_dm_step, [bs])
        noisy_x, sigma_t = forward_diffusion(x, timestep)

        with (torch.no_grad()):
            pred_fake_image = mu_fake(noisy_x, sigma_t)
            pred_real_image = mu_real(noisy_x, sigma_t)

        weighting_factor = torch.abs(x - pred_real_image).mean(dim=[1, 2, 3], keepdim=True)
        grad = (pred_fake_image - pred_real_image) / weighting_factor
        diff = (x - grad).detach()  # stop-gradient
        return 0.5 * F.mse_loss(x, diff, reduction=self.reduction)


class DenoisingLoss(_Loss):
    """
    Loss function for DMD (Algorithm 3) proposed in
    "One-step Diffusion with Distribution Matching Distillation".
    """

    def forward(self, pred_fake_image: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.mean(weight[:, None, None, None] * (pred_fake_image - x) ** 2)
