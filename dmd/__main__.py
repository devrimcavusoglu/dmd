import fire

from dmd.dataset.dataset_generator import generate_distillation_dataset
from dmd.generate import EDMGenerator
from dmd.train import run

if __name__ == "__main__":
    fire.Fire(
        {
            "generate-edm": EDMGenerator,
            "generate-dataset": generate_distillation_dataset,
            "train": run,
        }
    )
