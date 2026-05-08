import argparse
import csv
import datetime
import json
import os
import sys

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from unidbo_core.data.dataset import WaymaxDataset
from unidbo_core.model.utils import set_seed


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _pick_metric(metrics: dict, candidates: list[str]) -> float:
    for k in candidates:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                continue
    return float("nan")


def _resolve_runtime(args):
    accelerator = "gpu" if (str(args.device).lower().startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if args.accelerator is not None and str(args.accelerator).lower() != "auto":
        accelerator = str(args.accelerator).lower()

    if accelerator == "gpu" and not torch.cuda.is_available():
        print("[OpenLoop] CUDA unavailable, fallback to CPU.")
        accelerator = "cpu"

    if args.num_gpus is not None:
        req_devices = int(args.num_gpus)
    elif args.devices is not None:
        req_devices = int(args.devices)
    else:
        req_devices = 1 if accelerator == "cpu" else -1

    if accelerator == "cpu":
        devices = 1
        strategy = "auto"
        precision = "32-true"
        return accelerator, devices, strategy, precision

    n_gpu = torch.cuda.device_count()
    if req_devices == -1:
        devices = n_gpu
    else:
        devices = req_devices

    if devices <= 0:
        raise ValueError("For GPU run, --devices/--num_gpus must be >=1 or -1.")
    if devices > n_gpu:
        raise ValueError(f"Requested devices={devices}, but only {n_gpu} GPUs are visible.")

    strategy = DDPStrategy(find_unused_parameters=False) if devices > 1 else "auto"
    precision = args.precision if args.precision is not None else "bf16-mixed"
    return accelerator, devices, strategy, precision


@torch.no_grad()
def evaluate(args):
    set_seed(int(args.seed))
    pl.seed_everything(int(args.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("hyper_parameters", {}).get("cfg", None)
    if not isinstance(ckpt_cfg, dict):
        raise RuntimeError("Checkpoint does not contain hyper_parameters.cfg")

    # Open-loop protocol: validate CS branch only.
    model_cfg = dict(ckpt_cfg)
    model_cfg["branch_mode"] = "cs"
    model_cfg["branch_select_mode"] = "cs"
    model_cfg["train_dhn"] = False
    model_cfg["train_cs"] = True

    from unidbo_model import UniDBOModel

    model = UniDBOModel.load_from_checkpoint(
        args.model_path,
        map_location="cpu",
        strict=False,
        cfg=model_cfg,
    )
    model.eval()

    dataset = WaymaxDataset(args.val_data_path)
    if len(dataset) <= 0:
        raise ValueError(f"No validation scenes found in: {args.val_data_path}")

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    accelerator, devices, strategy, precision = _resolve_runtime(args)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if args.max_batches is None else int(args.max_batches),
        precision=precision,
    )

    out = trainer.validate(model, dataloaders=loader, verbose=False)
    metrics = out[0] if isinstance(out, list) and len(out) > 0 else {}

    ade = _pick_metric(metrics, ["val/ADE", "val/ADE_epoch"])
    fde = _pick_metric(metrics, ["val/FDE", "val/FDE_epoch"])
    loss = _pick_metric(metrics, ["val/loss", "val/loss_epoch"])

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, "open_loop_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["ADE", f"{ade:.6f}"])
        w.writerow(["FDE", f"{fde:.6f}"])
        w.writerow(["loss", f"{loss:.6f}"])

    summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": os.path.abspath(args.model_path),
        "val_data_path": os.path.abspath(args.val_data_path),
        "output_dir": os.path.abspath(args.output_dir),
        "seed": int(args.seed),
        "device": accelerator,
        "devices": int(devices) if isinstance(devices, int) else str(devices),
        "precision": str(precision),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_batches": None if args.max_batches is None else int(args.max_batches),
        "branch_mode": "cs",
        "branch_select_mode": "cs",
        "metrics": {
            "ADE": ade,
            "FDE": fde,
            "loss": loss,
        },
    }
    summary_json = os.path.join(args.output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Done] metrics_csv={metrics_csv}")
    print(f"[Done] summary_json={summary_json}")
    print(f"ADE: {ade:.6f}")
    print(f"FDE: {fde:.6f}")
    print(f"Loss: {loss:.6f}")
    print(
        f"Runtime: accelerator={accelerator}, devices={devices}, "
        f"strategy={strategy}, precision={precision}, batch_size={int(args.batch_size)}, "
        f"num_workers={int(args.num_workers)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/open_loop.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None, help="Shorthand for --accelerator gpu --devices N")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="results/train_logs/<RUN_DIR>/checkpoints/best-epoch=XX.ckpt")
    parser.add_argument("--val_data_path", type=str, default="data/val")
    parser.add_argument("--output_dir", type=str, default="results/open_loop_eval")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        cfg_args = load_yaml(args.config)
        defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
        for key, value in cfg_args.items():
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    evaluate(args)


if __name__ == "__main__":
    main()
