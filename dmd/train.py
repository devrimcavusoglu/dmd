import datetime
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from neptune import Run
from torch.backends import cudnn
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dmd import NEPTUNE_CONFIG_PATH
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.loss import DenoisingLoss, GeneratorLoss
from dmd.modeling_utils import load_model
from dmd.training.training_loop import train_one_epoch
from dmd.utils.common import create_experiment, set_seed
from dmd.utils.training import is_main_process, save_on_master

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from fvcore.nn import FlopCountAnalysis, flop_count, flop_count_table, parameter_count
    from utils import sfc_flop_jit

    has_fvcore = True
except ImportError:
    has_fvcore = False


def train(
    generator: torch.nn.Module,
    mu_fake: torch.nn.Module,
    mu_real: torch.nn.Module,
    data_loader_train: DataLoader,
    loss_g: TorchLoss,
    loss_d: TorchLoss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str,
    max_norm: float = 10,
    amp_autocast=None,
    neptune_run: Optional[Run] = None,
    cudnn_benchmark: bool = True,
    is_distributed: bool = False,
):
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    if cudnn_benchmark:
        cudnn.benchmark = True

    for epoch in range(epochs):
        if is_distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            generator,
            mu_fake,
            mu_real,
            data_loader_train,
            loss_g,
            loss_d,
            optimizer_g,
            optimizer_d,
            device,
            epoch,
            max_norm=max_norm,
            amp_autocast=amp_autocast,
            neptune_run=neptune_run,
            output_dir=output_dir,
        )

        # lr_scheduler.step(epoch)
        if output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                save_on_master(
                    {
                        "model_g": generator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "model_d": mu_fake.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        # "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        # "model_ema": get_state_dict(model_ema),
                        # "args": args,
                    },
                    checkpoint_path,
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            # **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def run(
    model_path: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int = 56,
    num_workers: int = 10,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    dmd_loss_timesteps: int = 1000,
    dmd_loss_lambda: float = 0.25,
    device: str = None,
    log_neptune: bool = False,
    cudnn_benchmark: bool = True,
    amp_autocast: Optional = None,
    max_norm: float = 10.0,
) -> None:
    """
    Starts the training phase.

    Args:
        model_path (str): Path to the model.
        data_path (str): Path of the h5 dataset file.
        output_dir (str): Path to the output directory to save the model.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size used in training process. [default: 64]
        num_workers (int): Number of workers for data loader. [default: 10]
        lr (float): Learning rate. [default: 5e-5]
        weight_decay (float): Weight decay for optimizer. [default: 0.01]
        betas (tuple(float, float)): Beta parameters for the optimizer. [default: (0.9, 0.999)]
        dmd_loss_timesteps (int): Number of timesteps to use for DMD loss. [default: 1000]
        dmd_loss_lambda (float): Lambda for the DMD loss. [default: 0.25]
        device (Optional(str)): Device to run the models on. [default: None]
        log_neptune (bool): Whether to log metrics to neptune. [default: False]
        cudnn_benchmark (bool): Whether to use CUDNN benchmark. [default: True]
        amp_autocast (Optional): Whether to use AMP autocast. [default: None]
        max_norm (Optional[float]): Maximum norm of the gradients. [default: 10.0]
    """
    # Prepare dataloader
    data_path = Path(data_path).resolve()
    training_dataset = CIFARPairs(data_path)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    mu_real = load_model(network_path=model_path, device=device)
    mu_fake = load_model(network_path=model_path, device=device)
    generator = load_model(network_path=model_path, device=device)

    # Create losses
    generator_loss = GeneratorLoss(timesteps=dmd_loss_timesteps, lambda_reg=dmd_loss_lambda)
    diffusion_loss = DenoisingLoss()

    # Create optimizers
    generator_optimizer = AdamW(params=generator.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    neptune_run = None
    if log_neptune:
        # create neptune run
        neptune_run = create_experiment(NEPTUNE_CONFIG_PATH)

        neptune_run["training_args"] = {
            "model_path": model_path,
            "data_path": data_path,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": "lr",
            "weight_decay": weight_decay,
            "betas": betas,
            "dmd_loss_timesteps": dmd_loss_timesteps,
            "dmd_loss_lambda": dmd_loss_lambda,
            "device": device,
            "cudnn_benchmark": cudnn_benchmark,
            "amp_autocast": amp_autocast,
            "max_norm": max_norm,
        }

    # start training
    train(
        generator=generator,
        mu_real=mu_real,
        mu_fake=mu_fake,
        data_loader_train=train_loader,
        device=device,
        loss_g=generator_loss,
        loss_d=diffusion_loss,
        optimizer_g=generator_optimizer,
        optimizer_d=diffuser_optimizer,
        epochs=epochs,
        neptune_run=neptune_run,
        output_dir=output_dir,
        cudnn_benchmark=cudnn_benchmark,
        amp_autocast=amp_autocast,
        max_norm=max_norm,
    )

    if neptune_run:
        neptune_run.stop()
