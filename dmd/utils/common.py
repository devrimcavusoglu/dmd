import math
import os
import random
from configparser import ConfigParser
from typing import Optional

import neptune
import numpy as np
import PIL.Image
import torch

from dmd.utils.array import torch_to_pillow
from dmd.utils.training import get_rank


def seed_everything(seed):
    seed = seed + get_rank()
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_cfg(fp: str, encoding: Optional[str] = None, **kwargs) -> ConfigParser:
    """
    Reads a config from given filepath.
    """
    cfg = ConfigParser(**kwargs)
    cfg.read(fp, encoding=encoding)
    return cfg


def create_experiment(config_path: str) -> neptune.Run:
    """
    Creates a Neptune Experiment
    """
    cfg = read_cfg(config_path)["credentials"]
    return neptune.init_run(project=cfg["project"], api_token=cfg["token"])


def image_grid(imgs, rows, cols, margin=2):
    """
    Taken from https://stackoverflow.com/a/65583584/7871601
    """
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new("RGB", (cols * w + (cols + 1) * margin, rows * h + (rows + 1) * margin), "WHITE")
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * (w + margin) + margin, i // cols * (h + margin) + margin))
    return grid


def closest_divisors(n: int):
    """
    Helper function for finding closest integer divisors.
    Taken from https://stackoverflow.com/a/40700741
    """
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a


def display_samples(samples: torch.Tensor):
    pims = torch_to_pillow(samples)
    n_row, n_col = closest_divisors(len(samples))
    return image_grid(pims, n_row, n_col)
