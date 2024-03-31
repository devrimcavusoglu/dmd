import fire

from dmd.edm.generate import generate

if __name__ == "__main__":
    fire.Fire(
        {
            "generate-edm": generate,
        }
    )
