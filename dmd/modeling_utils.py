import pickle
import sys

import torch
from torch.nn import Module

from dmd import SOURCES_ROOT, dnnlib
from dmd.torch_utils import distributed as dist


def load_model(network_path: str, device: torch.device) -> None:
    """
    Loads a pretrained model from given path. This function loads the model from the original
    pickle file of EDM models.

    Note:
        The saved binary files (pickle) contain serialized Python object where some of them
        require certain imports. This module is not identical to the structure of 'NVLabs/edm',
        see the related comment.

    Args:
        network_path (str): Path to the model weights.
        device (torch.device): Device to load the model to.
    """
    # Refactoring the package structure and import scheme (e.g. this module) breaks the loading of the
    # pickle file (as it also possesses the complete module structure at the save time). The following
    # line is a little trick to make the import structure the same to load the pickle without a failure.
    sys.path.insert(0, SOURCES_ROOT.as_posix())
    with dnnlib.util.open_url(network_path, verbose=(dist.get_rank() == 0)) as f:
        return pickle.load(f)["ema"].to(device)


def copy_weights(original: Module, clone: Module) -> None:
    """
    Copies weights from `original` to `clone`.
    """
    clone.load_state_dict(original.state_dict())
