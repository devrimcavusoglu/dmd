from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from dmd.dataset.cifar_pairs import CIFARPairs


def run(data_path: str, batch_size: int = 64) -> None:
    """
    Starts the training phase.

    Args:
        data_path (str): Path of the h5 dataset file.
        batch_size (int): Batch size used in training process. [default: 64]
    """
    data_path = Path(data_path).resolve()

    training_dataset = CIFARPairs(data_path)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    for _ in tqdm(training_dataloader):
        pass
        # print(data["instance_id"]) # Example usage
        # Available keys: (instance_id, image, latent, class_id, seed)
        # break  # For testing purpose
