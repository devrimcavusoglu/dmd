import fire

from dmd.dataset_generator import generate_distillation_dataset
from dmd.edm.generate import EDMGenerator

if __name__ == "__main__":
    fire.Fire(
        {
            "generate-edm": EDMGenerator,
            "generate-dataset": generate_distillation_dataset
        }
    )
