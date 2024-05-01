import torch
import torch.nn as nn
import numpy as np
from dmd.modeling_utils import encode_labels
from torchvision.models.inception import inception_v3
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from tqdm import tqdm

class FID():
    def __init__(self, data_loader, device="cuda") -> None:
        self.dataloader = data_loader
        self.device = device
        self.inception_model = inception_v3(pretrained=True, transform_input=True).to(self.device)
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.resize_images = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(self.device)

    def get_inception_features(self, image_batch):
        image_batch = self.resize_images(image_batch)
        inception_output = self.inception_model(image_batch)
        return inception_output.data.cpu().numpy()
    
    @torch.no_grad()
    def __call__(self, generator):
        inception_feature_batches_fake = []
        for image, class_idx in tqdm(self.dataloader, desc=f'FID - Fake Data Feature Extraction', total=len(self.dataloader)):
            z = torch.randn_like(image, device=self.device)
            # Scale Z ~ N(0,1) (z and z_ref) w/ 80.0 to match the sigma_t at T_n
            z = z * 80.0
            class_idx = class_idx.to(self.device, non_blocking=True)
            class_ids = encode_labels(class_idx, generator.label_dim)
            sigmas = torch.tensor([80.0] * z.shape[0], device=self.device)
            fake_image_batch = generator(z, sigmas, class_labels=class_ids)
            fake_image_batch = (fake_image_batch + 1) / 2.0 # Normalizing the pixel values from [-1,1] to [0,1]
            inception_feature_batch = self.get_inception_features(fake_image_batch)
            inception_feature_batches_fake.append(inception_feature_batch)
        inception_features_fake = np.concatenate(inception_feature_batches_fake)

        inception_feature_batches_real = []
        for image_batch, _ in tqdm(self.dataloader, desc=f'FID - Real Data Feature Extraction', total=len(self.dataloader)):
            image_batch = image_batch.to(self.device, non_blocking=True).to(torch.float32)
            inception_feature_batch = self.get_inception_features(image_batch)
            inception_feature_batches_real.append(inception_feature_batch)
        inception_features_real= np.concatenate(inception_feature_batches_real)
        
        mu_fake, sigma_fake = inception_features_fake.mean(axis=0), cov(inception_features_fake, rowvar=False)
        mu_real, sigma_real = inception_features_real.mean(axis=0), cov(inception_features_real, rowvar=False)
        ssdiff = np.sum((mu_fake - mu_real) ** 2.0)
        cov_mean = sqrtm(sigma_fake.dot(sigma_real))
        if iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        frechet_inception_distance = ssdiff + trace(sigma_fake + sigma_real - 2.0 * cov_mean)
        return frechet_inception_distance



