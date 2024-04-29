import h5py
from torch.utils.data import Dataset


class CIFARPairs(Dataset):
    _shape = (3, 32, 32)  # CHW

    def __init__(self, h5_dataset_path):
        self.h5_dataset_path = h5_dataset_path
        self.dataset = None
        with h5py.File(self.h5_dataset_path, "r") as file:
            self.num_samples = len(file["data"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Dataset is created here to avoid errors if (number of workers > 1) in dataloader
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_dataset_path, "r")["data"]

        sample = self.dataset[str(index)]
        pairs = sample[()]
        attributes = sample.attrs
        image, latent = pairs

        return_dict = {
            "instance_id": index,
            "image": image,
            "latent": latent,
            "class_id": attributes["class_idx"],
            "seed": attributes["seed"],
        }

        return return_dict

    @property
    def image_shape(self):
        return list(self._shape)

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]
