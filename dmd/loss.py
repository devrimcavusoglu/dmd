import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss, MSELoss

from dmd.sampler import edm_sampler, get_sigmas_karras
from dmd.utils.array import torch_to_pillow


class DistributionMatchingLoss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse_loss = MSELoss()

    def forward(self, mu_real: Module, mu_fake: Module, x: torch.Tensor, min_dm_step: int = 10, max_dm_step: int = 100) -> torch.Tensor:
        bs = x.shape[0]
        timestep = torch.randint(min_dm_step, max_dm_step, [bs])
        noise = torch.randn_like(x, device=x.device)

        sigma = get_sigmas_karras(1000, sigma_min=0.002, sigma_max=80, rho=7.0, device=x.device)
        print("Timestep:", timestep)
        ns = noise * sigma[-timestep[0], None, None, None]  # broadcast for scalar product
        noisy_x = x + ns
        torch_to_pillow(x, 0).show()
        torch_to_pillow(ns, 0).show()
        torch_to_pillow(noisy_x, 0).show()

        with torch.no_grad():
            pred_fake_image = edm_sampler(mu_fake, noisy_x, steps=timestep[0])
            pred_real_image = edm_sampler(mu_real, noisy_x, steps=timestep[0])

        torch_to_pillow(pred_fake_image, 0).show()
        # torch_to_pil(pred_real_image, 0).show()
        print("Is pred/real equal ?", torch.all(torch.eq(pred_fake_image, pred_real_image)))
        print("Is noisy_x/real equal ?", torch.all(torch.eq(noisy_x, pred_real_image)))
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
    import numpy as np

    from dmd.modeling_utils import load_model

    loss = DistributionMatchingLoss()
    device = torch.device("cuda")
    im1, lt1 = np.load("/home/devrim/lab/gh/dmd/data/distillation_dataset/samples/000000.npy")
    im2, lt2 = np.load("/home/devrim/lab/gh/dmd/data/distillation_dataset/samples/000003.npy")
    print(im1.shape, lt1.shape)
    images = torch.from_numpy(np.stack([im1, im2], axis=0)).to(device)
    latents = torch.from_numpy(np.stack([lt1, lt2], axis=0)).to(device)
    # numpy_to_pil(im1).show()
    # numpy_to_pil(im2).show()
    # numpy_to_pil(lt1).show()
    # numpy_to_pil(lt2).show()
    # x = torch.randn(16, 3, 32, 32, device=device)  # B,C,W,H
    mu_real = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", device=device)
    mu_fake = load_model(network_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
                         device=device)
    l = loss(mu_real, mu_fake, images)
    print(l.item())
