# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
from typing import Optional

import psutil
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dmd import dnnlib
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.loss import GeneratorLoss, DenoisingLoss
from dmd.modeling_utils import load_model
from dmd.torch_utils import distributed as dist
from dmd.torch_utils import training_stats
from dmd.torch_utils import misc


def training_loop(
    data_path: str,                 # Options for training set.
    network_path: str,              # Network(s) used in training.
    run_dir: str,                   # Output directory.
    data_loader_kwargs: dict = None,    # Options for torch.utils.data.DataLoader.
    optimizer_kwargs    = {},           # Options for optimizer.
    seed: int                = 0,            # Global random seed.
    batch_size: int          = 512,          # Total batch size for one training iteration.
    batch_gpu: Optional[int] = None,         # Limit batch size per GPU, None = no limit.
    total_kimg: int          = 200000,       # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,          # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,         # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,        # Learning rate ramp-up duration.
    loss_scaling        = 1,            # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,           # Interval of progress prints.
    snapshot_ticks      = 50,           # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,          # How often to dump training state, None = disable.
    resume_pkl          = None,         # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,         # Start from the given training state, None = reset training state.
    resume_kimg         = 0,            # Start from the given training progress.
    cudnn_benchmark     = True,         # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    """
    Main training loop.

    TODO: Add gradient clipping (L2 norm of 10).

    Args:
        data_path (str): path to data directory.
        network_path (str): path to network file.
        run_dir (str): path to run directory.
        data_loader_kwargs (dict): keyword arguments to pass to DataLoader.
    """
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = CIFARPairs(data_path)
    # dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(DataLoader(dataset=dataset_obj, shuffle=True, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels)
    mu_real = load_model(network_path=network_path, device=device)
    mu_fake = load_model(network_path=network_path, device=device)
    generator = load_model(network_path=network_path, device=device)
    mu_real.eval().requires_grad_(False)
    mu_fake.train().requires_grad_(True).to(device)
    generator.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    generator_loss = GeneratorLoss()
    diffusion_loss = DenoisingLoss()

    generator_optimizer = AdamW(params=generator.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999))

    ddp_generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[device])
    ddp_fake = torch.nn.parallel.DistributedDataParallel(mu_fake, device_ids=[device])
    ddp_real = torch.nn.parallel.DistributedDataParallel(mu_real, device_ids=[device])

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        generator_optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_generator, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')


if __name__ == "__main__":
    training_loop()
