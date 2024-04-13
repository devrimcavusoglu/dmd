import numpy as np
import h5py
import json
import os
from tqdm import tqdm

dataset_root = "data/distillation_dataset"
annotations_dir = os.path.join(dataset_root, "annotations.json")
with open(annotations_dir) as file:
    annotations_dict = json.load(file)
num_samples = len(annotations_dict["data"])

h5_dataset_folder =  os.path.join("data", "distillation_dataset_h5")
os.makedirs(h5_dataset_folder, exist_ok=True)
h5_dataset_path = os.path.join(h5_dataset_folder, "cifar.hdf5")
hf = h5py.File(h5_dataset_path, 'w')

for i in tqdm(range(num_samples)):
    sample_dict = annotations_dict["data"][i]
    class_idx = sample_dict["class_idx"]
    pairs_path = sample_dict["pairs_path"]
    seed = sample_dict["seed"]
    iid = sample_dict["iid"]
    pairs = np.load(os.path.join(dataset_root, pairs_path))

    h5_sample_path = f"/data/{iid}"
    hf[h5_sample_path] = pairs
    hf[h5_sample_path].attrs['class_idx'] = class_idx
    hf[h5_sample_path].attrs['seed'] = seed
    
hf.close()