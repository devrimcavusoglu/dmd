import torch
import torch.nn.functional as F
from piq import LPIPS
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torchvision.transforms import Resize

from dmd.modeling_utils import forward_diffusion


class DistributionMatchingLoss(_Loss):
    """
    Loss function for DMD (Algorithm 2) proposed in
    "One-step Diffusion with Distribution Matching Distillation".
    """

    def __init__(self, timesteps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timesteps = timesteps

    def forward(self, mu_real: Module, mu_fake: Module, x: torch.Tensor, class_ids: torch.Tensor = None) -> torch.Tensor:
        b, c, w, h = x.shape

        # In practice T_min, T_max choices follows DreamFusion as follows
        T_min, T_max = int(0.02 * self.timesteps), int(0.98 * self.timesteps)
        timestep = torch.randint(T_min, T_max, [b])
        noisy_x, sigma_t = forward_diffusion(x, timestep)

        with (torch.no_grad()):
            pred_fake_image = mu_fake(noisy_x, sigma_t, class_labels=class_ids)
            pred_real_image = mu_real(noisy_x, sigma_t, class_labels=class_ids)

        weighting_factor = torch.abs(x - pred_real_image).mean(
            dim=[1, 2, 3], keepdim=True
        )  # /  (sigma_t**2)  # Eqn. 8
        grad = (pred_fake_image - pred_real_image) / weighting_factor
        diff = (x - grad).detach()  # stop-gradient
        return 0.5 * F.mse_loss(x, diff, reduction=self.reduction)


class GeneratorLoss(_Loss):
    def __init__(self, timesteps: int = 1000, lambda_reg: float = 0.25, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self.dmd_loss = DistributionMatchingLoss(timesteps)
        self.lpips = LPIPS()
        self.lambda_reg = lambda_reg

    def forward(
        self, mu_real: Module, mu_fake: Module, x: torch.Tensor, x_ref: torch.Tensor, y_ref: torch.Tensor,
            class_ids: torch.Tensor = None
    ) -> torch.Tensor:
        loss_kl = self.dmd_loss(mu_real, mu_fake, x, class_ids)

        # Apply preprocessing
        x_ref = (x_ref + 1) / 2.0
        y_ref = (y_ref + 1) / 2.0
        transform = Resize(224)
        x_ref = transform(x_ref)
        y_ref = transform(y_ref)
        loss_reg = self.lpips(x_ref, y_ref)
        return loss_kl + self.lambda_reg * loss_reg


class DenoisingLoss(_Loss):
    """
    Loss function for DMD (Equation 6 / Algorithm 3) proposed in
    "One-step Diffusion with Distribution Matching Distillation".
    """

    def forward(self, mu_fake: Module, x: torch.Tensor, t: torch.Tensor, class_ids: torch.Tensor = None) -> torch.Tensor:
        x_t, sigma_t = forward_diffusion(x.detach(), t)  # stop grad
        # Algorithm SNR + 1 / sigma_data^2 for EDM (sigma_data = 0.5)
        pred_fake_image = mu_fake(x_t, sigma_t, class_labels=class_ids)
        weight = 1 / sigma_t**2 + 1 / mu_fake.sigma_data**2
        return torch.mean(weight[:, None, None, None] * (pred_fake_image - x.detach()) ** 2)
