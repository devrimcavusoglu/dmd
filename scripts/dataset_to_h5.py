import json
from pathlib import Path
from typing import Union

import h5py
import numpy as np
from tqdm import tqdm


def convert_json_to_h5(data_path: Union[str, Path]) -> None:
    """
    Convert json file to h5.

    Args:
        data_path (str): Path to the dataset directory (COCO format).
    """

    data_path = Path(data_path)
    annotations_dir = data_path / "annotations.json"
    with open(annotations_dir) as file:
        annotations_dict = json.load(file)
    num_samples = len(annotations_dict["data"])

    h5_dataset_folder = data_path.parent / f"{data_path.stem}_h5"
    h5_dataset_folder.mkdir(exist_ok=True)
    h5_dataset_path = h5_dataset_folder / "cifar.hdf5"
    hf = h5py.File(h5_dataset_path, 'w')

    for i in tqdm(range(num_samples)):
        sample_dict = annotations_dict["data"][i]
        class_idx = sample_dict["class_idx"]
        pairs_path = sample_dict["pairs_path"]
        seed = sample_dict["seed"]
        iid = sample_dict["iid"]
        pairs = np.load(data_path / pairs_path)

        h5_sample_path = f"/data/{iid}"
        hf[h5_sample_path] = pairs
        hf[h5_sample_path].attrs['class_idx'] = class_idx
        hf[h5_sample_path].attrs['seed'] = seed

    hf.close()


if __name__ == "__main__":
    from utils import DATA_DIR
    data_path = DATA_DIR / "distillation_dataset"
    convert_json_to_h5(data_path=data_path)
