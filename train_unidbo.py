import argparse
import datetime
import os
import sys

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from unidbo_core.data.dataset import WaymaxDataset


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict):
    pl.seed_everything(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")

    train_shuffle = bool(cfg.get("train_shuffle", False))
    if train_shuffle:
        raise ValueError("UniDBO release expects train_shuffle=false for reproducible data order.")

    output_root = cfg.get("log_dir", "results/train_logs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.get('model_name', 'UniDBO')}_{timestamp}"
    output_path = os.path.join(output_root, run_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_dataset = WaymaxDataset(data_dir=cfg["train_data_path"])
    train_count = cfg.get("train_sample_count", None)
    if train_count is not None:
        keep_n = min(int(train_count), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(keep_n)))
        print(f"[TrainData] restricted train samples to {keep_n}")

    val_dataset = WaymaxDataset(data_dir=cfg["val_data_path"])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=True,
    )

    from unidbo_model import UniDBOModel

    model = UniDBOModel(cfg=cfg)
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        remapped = {}
        for key, value in state_dict.items():
            if ("feat_" + "kd_") in key:
                continue
            legacy_dhn_prefix = "deno" + "iser."
            legacy_cs_prefix = "c" + "t_deno" + "iser."
            if key.startswith(legacy_dhn_prefix):
                key = "dhn_branch." + key[len(legacy_dhn_prefix) :]
            elif key.startswith(legacy_cs_prefix):
                key = "cs_branch." + key[len(legacy_cs_prefix) :]
            elif key == "c" + "t_sigma_levels":
                key = "cs_sigma_levels"
            remapped[key] = value
        model.load_state_dict(remapped, strict=False)
        print(f"Loaded weights from {ckpt_path}")

    loggers = [
        CSVLogger(output_path, name="csv"),
        TensorBoardLogger(save_dir=output_path, name="tb"),
    ]
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_path, "checkpoints"),
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            filename="best-epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(output_path, "checkpoints"),
            save_top_k=-1,
            monitor=None,
            filename="epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    available_gpus = torch.cuda.device_count()
    requested_accelerator = cfg.get("accelerator", "auto")
    accelerator = "gpu" if requested_accelerator == "auto" and available_gpus > 0 else requested_accelerator
    if accelerator == "auto":
        accelerator = "cpu"
    if accelerator == "gpu" and available_gpus == 0:
        print("[Trainer] CUDA is unavailable; using CPU.")
        accelerator = "cpu"
    devices = cfg.get("devices", -1 if accelerator == "gpu" else 1)
    if accelerator == "cpu" and (devices == -1 or int(devices) < 1):
        devices = 1
    strategy = DDPStrategy(find_unused_parameters=True) if accelerator == "gpu" and available_gpus > 1 else "auto"
    precision = cfg.get("precision", "bf16-mixed" if accelerator == "gpu" else "32-true")
    if accelerator == "cpu" and precision != "32-true":
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=int(cfg["epochs"]),
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(cfg.get("log_every_n_steps", 1)),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=int(cfg.get("gradient_accumulation_steps", 1)),
    )

    print(
        "Training UniDBO: "
        f"branch_mode={cfg.get('branch_mode', 'both')}, "
        f"DHN={cfg.get('train_dhn', True)}, CS={cfg.get('train_cs', True)}, "
        f"accelerator={accelerator}, devices={devices}, precision={precision}"
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.get("init_from"))
    print(f"Done. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    # torchrun compatibility (PyTorch 2.x passes --local-rank).
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-cfg", "--config", type=str, default="configs/train.yaml")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--num_gpus", type=int, help="Shorthand for --accelerator gpu --devices N")
    parser.add_argument("--accelerator", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_sample_count", type=int)
    parser.add_argument("--branch_mode", type=str, choices=["dhn", "cs", "both"])
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_gpus is not None:
        if int(args.num_gpus) < 1:
            raise ValueError("--num_gpus must be >= 1")
        cfg["accelerator"] = "gpu"
        cfg["devices"] = int(args.num_gpus)
    for key, value in vars(args).items():
        if key not in {"config", "num_gpus"} and value is not None:
            cfg[key] = value
    train(cfg)


if __name__ == "__main__":
    main()
