import argparse
import csv
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


def _resolve_runtime(args):
    available_gpus = torch.cuda.device_count()
    accelerator = str(args.accelerator).lower() if args.accelerator is not None else "auto"

    if accelerator == "auto":
        accelerator = "gpu" if available_gpus > 0 else "cpu"

    device_arg = str(args.device).lower()
    if device_arg.startswith("cuda"):
        accelerator = "gpu"
    elif device_arg == "cpu":
        accelerator = "cpu"

    if args.num_gpus is not None:
        if int(args.num_gpus) < 1:
            raise ValueError("--num_gpus must be >= 1")
        accelerator = "gpu"
        devices = int(args.num_gpus)
    else:
        devices = args.devices

    if accelerator == "gpu" and available_gpus == 0:
        print("[OpenLoop] CUDA is unavailable; using CPU.")
        accelerator = "cpu"

    if devices is None:
        devices = -1 if accelerator == "gpu" else 1
    if accelerator == "cpu" and (devices == -1 or int(devices) < 1):
        devices = 1

    use_multi_gpu = accelerator == "gpu" and ((devices == -1) or (int(devices) > 1))
    strategy = DDPStrategy(find_unused_parameters=False) if use_multi_gpu else "auto"

    precision = args.precision
    if precision is None:
        precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    if accelerator == "cpu" and precision != "32-true":
        precision = "32-true"

    return accelerator, devices, strategy, precision


def _scenario_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith(".pkl"):
        name = name[:-4]
    if name.startswith("scenario_"):
        name = name[len("scenario_") :]
    return name


@torch.no_grad()
def evaluate(args):
    set_seed(int(args.seed))
    from unidbo_model import UniDBOModel

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("hyper_parameters", {}).get("cfg", None)
    if not isinstance(ckpt_cfg, dict):
        raise RuntimeError("Checkpoint does not contain hyper_parameters.cfg")

    # Open-loop protocol must use CS branch only, same validation path as training.
    model_cfg = dict(ckpt_cfg)
    model_cfg["branch_mode"] = "cs"
    model_cfg["branch_select_mode"] = "cs"
    model_cfg["train_dhn"] = False
    model_cfg["train_cs"] = True

    model = UniDBOModel.load_from_checkpoint(
        args.model_path,
        map_location="cpu",
        strict=False,
        cfg=model_cfg,
    )
    model.eval()

    dataset = WaymaxDataset(args.val_data_path)
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
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if args.max_batches is None else int(args.max_batches),
    )

    results = trainer.validate(model, dataloaders=loader, verbose=False)
    metrics = results[0] if results else {}

    ade = float(metrics.get("val/ADE", 0.0))
    fde = float(metrics.get("val/FDE", 0.0))
    loss = float(metrics.get("val/loss", 0.0))

    os.makedirs(args.output_dir, exist_ok=True)

    # Save per-scenario metrics for debugging/analysis.
    runtime_device = torch.device("cuda:0" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    model = model.to(runtime_device)
    per_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(runtime_device.type == "cuda"),
    )
    per_scene_path = os.path.join(args.output_dir, "open_loop_metrics_per_scenario.csv")
    with open(per_scene_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "scenario_id", "ADE", "FDE", "loss"])
        for idx, batch in enumerate(per_loader):
            batch = model.batch_to_device(batch, runtime_device)
            _, log_dict = model.forward_and_get_loss(batch, prefix="val/")
            row_ade = float(log_dict.get("val/ADE", 0.0))
            row_fde = float(log_dict.get("val/FDE", 0.0))
            row_loss = float(log_dict.get("val/loss", 0.0))
            scenario_id = _scenario_id_from_path(dataset.data_list[idx]) if idx < len(dataset.data_list) else str(idx)
            writer.writerow([idx, scenario_id, f"{row_ade:.6f}", f"{row_fde:.6f}", f"{row_loss:.6f}"])
        writer.writerow([])
        writer.writerow(["MICRO_AVG", "__micro__", f"{ade:.6f}", f"{fde:.6f}", f"{loss:.6f}"])

    metrics_path = os.path.join(args.output_dir, "open_loop_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["ADE", f"{ade:.6f}"])
        writer.writerow(["FDE", f"{fde:.6f}"])
        writer.writerow(["loss", f"{loss:.6f}"])

    print(f"Open-loop results saved to: {metrics_path}")
    print(f"Per-scenario results saved to: {per_scene_path}")
    print(f"ADE: {ade:.6f}")
    print(f"FDE: {fde:.6f}")
    print(f"Loss: {loss:.6f}")
    print(
        f"Runtime: accelerator={accelerator}, devices={devices}, "
        f"precision={precision}, cs_only_validation=True"
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
