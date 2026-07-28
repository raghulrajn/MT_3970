# Experiments

This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration and [MLflow](https://mlflow.org/docs/latest/quickstart.html) for experiment tracking.

## Training

```bash
# Basic training
uv run main.py model=my_model dataset=ddacs

# Override parameters
uv run main.py model=my_model dataset=ddacs model.optimizer.learning_rate=0.0001

# Hyperparameter sweep (-m flag)
uv run main.py -m model.optimizer.learning_rate=0.001,0.0001,0.00001
```

## Training Configuration

`configs/training/` holds the `default`, `fast_debug`, and `final_run`
profiles (`training=<profile>`). Two settings worth knowing about:

**Precision** (`training.precision`, default `bf16-mixed`): bf16-mixed is
[recommended by Lightning](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html)
over plain fp16-mixed -- it has the same dynamic range as fp32, so it
doesn't need loss scaling and is far less prone to NaNs/overflow, while
still giving a speedup on modern GPUs. Override per-profile or via CLI
(`training.precision=32-true`) if you hit a compatibility issue; valid
values are `bf16-mixed`, `16-mixed`, `32-true`, `64-true`.

**Data fraction** (`training.data_fraction`, must be between `0.0` and
`1.0`, default `1.0`): fraction of the train/val/test data used each
epoch. Set it below 1.0 for a faster, still-representative training run
(as opposed to `fast_debug`'s fixed tiny batch counts, which are a
pipeline smoke test, not real training):

```bash
# Train on 10% of the data
uv run main.py training.data_fraction=0.1

# Full data (default)
uv run main.py training.data_fraction=1.0
```

For DDACS (`metadata.num_samples: 32466` in `configs/dataset/ddacs.yaml`,
split 25,973 / 3,246 / 3,247 train/val/test):
`data_fraction=1.0` uses all ~32,466 simulations, `data_fraction=0.1`
uses ~10% of each split (~3,247 total: ~2,597 train / ~325 val /
~325 test). The exact per-split count is the fraction applied
independently to each split's own size, floored to a whole number of
batches -- it is not an exact 10% of raw samples, just close to it. The
resolved fraction and approximate total sample count are logged at the
start of every training run.

Internally this sets Lightning's `limit_train_batches` /
`limit_val_batches` / `limit_test_batches` to the same fraction; an
explicit `limit_*_batches` value in a profile (like `fast_debug`) always
takes precedence over `data_fraction`. Passing a value outside `[0.0, 1.0]`
(or an `h5`/streaming modality with a non-`1.0` fraction) raises a clear
`MisconfigurationException` from Lightning before training starts.

`data_fraction` only reduces *how many simulation samples* are drawn each
epoch -- it never touches the geometric content of a sample. For the
`graph`/`mesh` modality in particular, each retained sample keeps its
full mesh (fixed 11,236 nodes / 44,520 edges for the blank, plus the
tool meshes) and the complete set of process parameters; nothing about
node/edge count or the 7 process parameters is subsampled.

## Viewing Results

```bash
uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db
```

Open <http://127.0.0.1:5000>. Sort by metrics to find your best run and note its **Run ID**.

## Evaluate and Register

```bash
# Evaluate on all datasets and register to Model Registry
uv run evaluate.py --run_id <run_id> --datasets all --register

# Validate submission (~1 minute)
uv run pytest tests/test_submission.py -v -m "not slow"
```

Each registration creates a new version under your thesis_id (e.g. `MT_2024_042`).

## Reproduce a Run

```bash
uv run python -m src.utils.reproduce <run_id> --info     # show run info
uv run python -m src.utils.reproduce <run_id> --dry-run   # preview command
uv run python -m src.utils.reproduce <run_id>              # reproduce
```

## Storage

All experiment data lives in `./mlruns/`:

```
mlruns/
├── mlflow.db              # SQLite database (metrics, params, tags)
└── <experiment_id>/
    └── <run_id>/
        └── artifacts/
            ├── config.yaml    # Full config for reproduction
            ├── checkpoints/   # Model checkpoints
            └── <model_name>/  # Logged PyTorch model
```

## Troubleshooting

**MLflow UI shows no experiments:** Check the URI prefix:
```bash
# Correct
uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db
```

**Config errors:** Validate with `uv run main.py --cfg job`
