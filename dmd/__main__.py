import fire

from dmd.edm.generate import EDMGenerator

if __name__ == "__main__":
    fire.Fire(
        {
            "generate-edm": EDMGenerator,
        }
    )
