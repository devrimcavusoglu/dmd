import io
import os
import time
from collections import Counter
from configparser import ConfigParser
from typing import Any
from typing import Counter as TypingCounter
from typing import List, Optional

import neptune
import torch
import torch.distributed as dist

try:
    from fvcore.nn.jit_handles import conv_flop_count, get_shape
except ImportError:
    has_fvcore = False


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        for i in range(10):
            try:
                torch.save(*args, **kwargs)
                break
            except:
                print("Saving model failed for {} times, will retry".format(i + 1))
                time.sleep(30.0)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# copy-paste from
# https://github.com/facebookresearch/fvcore/blob/166a030e093013a934642ca3744592a2e3de5ea2/fvcore/nn/jit_handles.py#L143-L157
# change input length assert
def sfc_flop_jit(inputs: List[Any], outputs: List[Any]) -> TypingCounter[str]:
    """
    Count flops for cycle FC.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    assert w_shape[-1] == 1 and w_shape[-2] == 1, w_shape

    # use a custom name instead of "_convolution"
    return Counter({"conv": conv_flop_count(x_shape, w_shape, out_shape)})
