from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dmd import NEPTUNE_CONFIG_PATH
from dmd.dataset.cifar_pairs import CIFARPairs
from dmd.utils.logging import create_experiment, init_distributed_mode


def train(
    model,
    data_loader_train,
    data_loader_val,
    criterion,
    optimizer,
    loss_scaler,
    neptune_run: Optional = None,
    *,
    model_ema,
    mixup_fn,
    amp_autocast,
    lr_scheduler,
    output_dir,
    model_without_ddp,
    dataset_val,
    n_parameters,
    args,
    device,
):
    # build MINE stuff
    dim_spatial = args.embedding_dim
    dim_channel = (args.input_size // args.patch_size) ** 2
    if args.distillation_type != "none":
        model_regulizer, mine_network, mine_optimizer, objective = build_mine(
            model, dim_spatial, dim_channel, device
        )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, mine_samples = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.distillation_type,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune == "",  # keep in eval mode during finetuning
            amp_autocast=amp_autocast,
            n_mine_samples=args.n_mine_samples,
            neptune_run=neptune_run,
        )
        # regularize with MINE
        if mine_samples is not None:
            mine_regularization(
                model,
                mine_network,
                model_regulizer,
                mine_optimizer,
                objective,
                mine_samples,
                neptune_run=neptune_run,
            )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        # "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(
            data_loader_val, model, device, amp_autocast=amp_autocast, neptune_run=neptune_run
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def main(args) -> None:
    neptune_run = None
    if args.log_neptune:
        neptune_run = create_experiment(config_path=NEPTUNE_CONFIG_PATH)
        neptune_run["arguments"] = {
            "model": args.model,
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "device": args.device,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "model_ema": args.model_ema,
            "optimizer": args.optimizer,
            "optimizer_eps": args.optimizer_eps,
            "optimizer_betas": args.optimizer_betas,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "weight_decay": args.weight_decay,
            "clip_grad": args.clip_grad,
            "warmup_lr": args.warmup_lr,
            "use_amp": args.use_amp
        }
    init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if not use_amp:  # args.amp: Default  use AMP
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
            args.apex_amp = False
        elif has_apex:
            args.native_amp = False
            args.apex_amp = True
        else:
            raise ValueError(
                "Warning: Neither APEX or native Torch AMP is available, using float32."
                "Install NVIDA apex or upgrade to PyTorch 1.6"
            )
    else:
        args.apex_amp = False
        args.native_amp = False
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        print(
            "Warning: Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    # fix the seed for reproducibility
    set_seed(args.seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train, data_loader_val = create_loaders(
        dataset_train, dataset_val, distributed=args.distributed
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = get_model(args)

    if args.flops:
        if not has_fvcore:
            print("Please install fvcore first for FLOPs calculation.")
        else:
            # Set model to evaluation mode for analysis.
            model_mode = model.training
            model.eval()
            fake_input = torch.rand(1, 3, 224, 224)
            flops_dict, *_ = flop_count(
                model, fake_input, supported_ops={"torchvision::deform_conv2d": sfc_flop_jit}
            )
            count = sum(flops_dict.values())
            model.train(model_mode)
            print("=" * 30)
            print("fvcore MAdds: {:.3f} G".format(count))

    # This part is not changed from DeiT, should be refactored for allMLP
    if args.finetune:
        model = prepare_for_finetune(model)

    model.to(device)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        print("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print("Using native Torch AMP. Training in mixed precision.")
    else:
        print("AMP not enabled. Training in float32.")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    print("=" * 30)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = create_criterion(args)

    teacher_models = None
    if args.distillation_type != "none":
        # assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_models = []
        if args.data_set == "CIFAR":
            for teacher in args.teacher_model:
                teacher_model = torch.hub.load(
                    "chenyaofo/pytorch-cifar-models", f"cifar100_{teacher}", pretrained=True
                )
                teacher_models.append(teacher_model)
        elif args.data_set == "INAT":
            pass
        else:  # IMNET
            for teacher in args.teacher_model:
                teacher_model = create_model(
                    teacher,
                    pretrained=True,
                )
                teacher_models.append(teacher_model)

        for teacher_model in teacher_models:
            teacher_model.to(device)
            teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    if args.distillation_type != "none":
        criterion = DistillationLoss(
            criterion,
            teacher_models,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
            run=neptune_run,
        )

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        / f"{args.data_set}_{args.model}_s{args.patch_size}_{args.input_size}_{current_time}"
    )
    if not args.eval:
        output_dir.mkdir(exist_ok=False, parents=True)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint_path = Path(args.resume) / "checkpoint.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        test_stats = evaluate(
            data_loader_val, model, device, amp_autocast=amp_autocast, neptune_run=neptune_run
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    train(
        model,
        device=device,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        criterion=criterion,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
        mixup_fn=mixup_fn,
        amp_autocast=amp_autocast,
        lr_scheduler=lr_scheduler,
        output_dir=output_dir,
        model_without_ddp=model_without_ddp,
        dataset_val=dataset_val,
        n_parameters=n_parameters,
        args=args,
        neptune_run=neptune_run,
    )
    if neptune_run:
        neptune_run.stop()


def run(data_path: str, batch_size: int = 64) -> None:
    """
    Starts the training phase.

    Args:
        data_path (str): Path of the h5 dataset file.
        batch_size (int): Batch size used in training process. [default: 64]
    """
    data_path = Path(data_path).resolve()

    training_dataset = CIFARPairs(data_path)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    data_path = "/home/devrim/lab/gh/ms/dmd/data/distillation_dataset_h5/cifar.hdf5"
    network_path = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
    device = torch.device("cuda")
    dataset = CIFARPairs(data_path)
    # TODO: BS -> 64
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    mu_real = load_model(network_path=network_path, device=device)
    mu_fake = load_model(network_path=network_path, device=device)
    generator = load_model(network_path=network_path, device=device)

    generator_loss = GeneratorLoss()
    diffusion_loss = DenoisingLoss()

    generator_optimizer = AdamW(params=generator.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    diffuser_optimizer = AdamW(params=mu_fake.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.999))
    for _ in tqdm(range(epoch)):

        train_one_epoch(generator, mu_fake, mu_real, dataloader, generator_loss, diffusion_loss, generator_optimizer,
                        diffuser_optimizer, device, 1)
