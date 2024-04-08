import json
from pathlib import Path

from dmd.edm.generate import EDMGenerator


def generate_distillation_dataset(
    model_path: str, output_dir: str, device: str = None, size_per_class: int = 10000, batch_size: int = 64
):
    edm_generator = EDMGenerator(network_path=model_path, device=device)
    output_dir = Path(output_dir) / "distillation_dataset"
    annotations = {"data": [], "version": "0.1"}
    seeds = list(range(size_per_class))
    instance_id = 0
    for cls in range(2):
        edm_generator(
            output_dir.as_posix(),
            seeds=seeds,
            class_idx=cls,
            batch_size=batch_size,
            save_format="all",
        )
        for seed in seeds:
            annotations["data"].append({
                "image_path": f"images/class_{cls}/{seed:06d}.png",
                "latent_path": f"latents/class_{cls}/{seed:06d}.png",
                "class_idx": cls,
                "seed": seed,
                "iid": instance_id
            })
            instance_id += 1
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)
