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

    def forward(self, mu_real: Module, mu_fake: Module, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.shape

        # In practice T_min, T_max choices follows DreamFusion as follows
        T_min, T_max = int(0.02 * self.timesteps), int(0.98 * self.timesteps)
        timestep = torch.randint(T_min, T_max, [b])
        noisy_x, sigma_t = forward_diffusion(x, timestep)

        with (torch.no_grad()):
            pred_fake_image = mu_fake(noisy_x, sigma_t)
            pred_real_image = mu_real(noisy_x, sigma_t)

        weighting_factor = torch.abs(x - pred_real_image).mean(
            dim=[1, 2, 3], keepdim=True
        )  # / (sigma_t**2) Eqn. 8
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
        self, mu_real: Module, mu_fake: Module, x: torch.Tensor, x_ref: torch.Tensor, y_ref: torch.Tensor
    ) -> torch.Tensor:
        loss_kl = self.dmd_loss(mu_real, mu_fake, x)

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

    def forward(self, mu_fake: Module, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Algorithm SNR + 1 / sigma_data^2 for EDM (sigma_data = 0.5)
        pred_fake_image = mu_fake(x, sigma)
        weight = 1 / sigma**2 + 1 / mu_fake.sigma_data**2
        return torch.mean(weight[:, None, None, None] * (pred_fake_image - x) ** 2)


if __name__ == "__main__":
    import numpy as np

    from dmd.modeling_utils import load_model

    loss = DistributionMatchingLoss(1000)
    device = torch.device("cuda")
    im1, lt1 = np.load("/home/devrim/lab/gh/ms/dmd/data/distillation_dataset/samples/000000.npy")
    im2, lt2 = np.load("/home/devrim/lab/gh/ms/dmd/data/distillation_dataset/samples/000003.npy")
    print(im1.shape, lt1.shape)
    images = torch.from_numpy(np.stack([im1, im2], axis=0)).to(device)
    latents = torch.from_numpy(np.stack([lt1, lt2], axis=0)).to(device)
    print(latents.min(), latents.max(), latents.std())
    # numpy_to_pil(im1).show()
    # numpy_to_pil(im2).show()
    # numpy_to_pil(lt1).show()
    # numpy_to_pil(lt2).show()
    # x = torch.randn(16, 3, 32, 32, device=device)  # B,C,W,H
    # mu_real = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", device=device)
    # # mu_fake = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
    # #                      device=device)
    # loss(mu_real, mu_real, images)
