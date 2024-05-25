"""
Misc functions, including distributed helpers. Part of this file is taken and adopted
from devrimcavusoglu/std repository. See the original file below
https://github.com/devrimcavusoglu/std/blob/main/std/utils.py
"""

import datetime
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from dmd.utils.training import is_dist_avail_and_initialized, is_main_process, save_on_master


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", neptune_run: Optional = None, is_train: bool = True):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.neptune_run = neptune_run
        self.is_train = is_train

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if self.is_train:
                self.log_neptune(k, v)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def log_neptune(self, key, val):
        """
        Logs to neptune if neptune_run is given, silently passes otherwise.
        """
        prefix = "train/" if self.is_train else "val/"
        name = prefix + key
        if self.neptune_run:
            self.neptune_run[name].append(val)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable)))


class CheckpointHandler:
    """
    Checkpoint manager for saving and loading from a checkpoint of trained models.
    """

    def __init__(self, checkpoint_dir: str, lower_is_better: bool = True, resume_from_checkpoint: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lower_is_better = lower_is_better
        self._metric_value = float("inf") if lower_is_better else -float("inf")
        self._best_epoch = None

    def is_better(self, metric):
        if self.lower_is_better:
            return metric < self._metric_value
        return metric > self._metric_value

    def save(self, model_dict: Dict[str, Any], stats: Dict[str, Any], metric: float, epoch: int) -> None:
        is_best = self.is_better(metric)
        # save the last checkpoint
        save_on_master(model_dict, self.checkpoint_dir / f"last_checkpoint.pt")

        if is_best:
            self._metric_value = metric
            self._best_epoch = epoch
            save_on_master(model_dict, self.checkpoint_dir / f"best_checkpoint.pt")

        stats["best_epoch"] = self._best_epoch if epoch > 0 else None
        stats["best_metric"] = self._metric_value
        if is_main_process():
            with (self.checkpoint_dir / "log.txt").open("a") as f:
                f.write(json.dumps(stats) + "\n")
