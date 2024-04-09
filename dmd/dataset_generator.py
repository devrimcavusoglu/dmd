import json
from pathlib import Path

from dmd.edm.generate import EDMGenerator


def generate_distillation_dataset(
    model_path: str, output_dir: str, device: str = None, size_per_class: int = 10000, batch_size: int = 64
):
    """
    Creates a dataset for distillation training. This dataset contains noise, image pairs generated from
    the given EDM model.

    For CIFAR-10 training, we use the following configuration:
        ```
        model_path="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
        output_dir="data/distillation_dataset",
        size_per_class=10000,
        batch_size=1024  # <-- w/ 24 GB of vRAM
        ```
    """
    edm_generator = EDMGenerator(network_path=model_path, device=device)
    output_dir = Path(output_dir)
    annotations = {"data": [], "version": "0.1"}
    seeds = list(range(size_per_class))
    instance_id = 0
    for cls in range(2):
        edm_generator(
            output_dir.as_posix(),
            seeds=seeds,
            class_idx=cls,
            batch_size=batch_size,
            save_format="pairs",
            save_start_idx=instance_id
        )
        for iid in range(instance_id, instance_id+len(seeds)):
            annotations["data"].append({
                "pairs_path": f"samples/{iid:06d}.npy",
                "class_idx": cls,
                "seed": instance_id % len(seeds),
                "iid": instance_id
            })
            instance_id += 1
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)
