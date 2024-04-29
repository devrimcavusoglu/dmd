import random
from configparser import ConfigParser
from typing import Optional

import neptune
import numpy as np
import torch

from dmd.utils.training import get_rank


def set_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
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
