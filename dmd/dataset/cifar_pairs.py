import h5py
from torch.utils.data import Dataset


class CIFARPairs(Dataset):
    def __init__(self, h5_dataset_path):
        self.h5_dataset_path = h5_dataset_path
        self.dataset = None
        with h5py.File(self.h5_dataset_path, 'r') as file:
            self.num_samples = len(file["data"])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Dataset is created here to avoid errors if (number of workers > 1) in dataloader
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_dataset_path, 'r')["data"]
        
        sample = self.dataset[str(index)]
        pairs = sample[()]
        attributes = sample.attrs
        image, latent = pairs

        return_dict = {
            "instance_id": index,
            "image": image,
            "latent": latent,
            "class_id": attributes['class_idx'],
            "seed": attributes['seed']
        }

        return return_dict
