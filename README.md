

This repository contains the UniDBO codebase for:

- Data conversion
- Joint training of DHN and CS branches
- Validation and open-loop prediction (CS branch)
- Waymax closed-loop simulation (DHN branch)

## Build Environment

```bash
conda create -n unidbo python=3.10 -y
conda activate unidbo
pip install --upgrade pip
pip install -r requirements.txt
```

## Get Dataset

Download Waymo Open Motion Dataset (WOMD) tf_example files:

- https://waymo.com/open/data/motion/

 Please use version V1.2 tf_example data to work with Waymax.

## Data Preparation

Convert training and validation tfrecord data into UniDBO pickle format:

```bash
python tools/prepare_data.py \
    --input_path /path/to/womd/train_tfrecord_dir \
    --output_dir data/train \
    --save_raw

python tools/prepare_data.py \
    --input_path /path/to/womd/val_tfrecord_dir \
    --output_dir data/val \
    --save_raw
```

`--save_raw` stores scenario raw bytes for downstream Waymax closed-loop evaluation.

## Training 

```bash
python train_unidbo.py \
    --config configs/train.yaml \
    --num_gpus 8 \
    --train_data_path data/train \
    --val_data_path data/val \
    --batch_size 8 \
    --epochs 15
```

Training logs and checkpoints are written to `results/train_logs/`.

## Testing

Set checkpoint path from your training output:

```bash
CKPT=results/train_logs/<RUN_DIR>/checkpoints/best-epoch=XX.ckpt
```

### Open-Loop Prediction (CS branch, val1 protocol)

```bash
python eval_open_loop.py \
    --config configs/open_loop.yaml \
    --model_path ${CKPT} \
    --val_data_path data/val \
    --output_dir results/open_loop_eval
```

This command uses the same validation path as training with CS-only evaluation and reports ADE/FDE in `results/open_loop_eval/open_loop_metrics.csv`.

### Waymax Closed-Loop Simulation (DHN branch)

```bash
python test_unidbo.py \
    --config configs/closed_loop.yaml \
    --model_path ${CKPT} \
    --test_path /path/to/womd/val_tfrecord_dir \
    --max_scenarios 1000 \
    --save_mode csv \
    --output_dir results/closed_loop_eval
```

This command runs closed-loop rollout and writes Waymax metrics to CSV files under `results/closed_loop_eval/`.

## Visualization Samples

The folder `examples/sample/` contains 8 visualization examples for quick inspection.

- `*.mp4`: rendered closed-loop videos (recommended for anonymous repository demos).
- `*_sim.pkl`: optional simulation states for programmatic replay/analysis.

For anonymous review, keeping `mp4` files is sufficient. Keep `pkl` files only if you want to support additional reproducibility analysis beyond video inspection.
