# HelperScripts

Standalone pipeline for exporting, compiling, and benchmarking `Model_ContactAware`
across multiple inference backends (eager, TorchScript, `torch.compile`, ONNX Runtime,
TensorRT) on the DDACS point-cloud dataset. All scripts write their output artifacts to
`ModelArtifacts/` at the repo root and can be run directly with `python` — they resolve
the repo root from their own file path, so no `PYTHONPATH` setup or `cd` is required.

## Contents

| File | Purpose |
|---|---|
| [utils.py](utils.py) | Shared helpers (paths, model loaders, input prep, metrics). Not run directly. |
| [model_save.py](model_save.py) | Export a Lightning checkpoint to a TorchScript module (`model_eager.pt`). |
| [compile.py](compile.py) | Compile the eager model with `torch.compile` (3 modes) and sanity-check accuracy. |
| [trt_build.py](trt_build.py) | Export ONNX (`model.onnx`) and build TensorRT engines (`model_fp32.trt`, `model_tf32.trt`). |
| [latency_profiling.py](latency_profiling.py) | Benchmark latency (mean/p50/p90) and capture profiler traces for each artifact. |
| [inference.py](inference.py) | Run every available backend on chosen test samples, report metrics, and save plots. |

## Prerequisites

- Run from the project's `uv` environment: `uv sync`, then either prefix commands with
  `uv run` or activate `.venv` first (`source .venv/bin/activate`).
- A CUDA GPU is required for `trt_build.py`, `latency_profiling.py`, and `inference.py`
  (they call `get_device(require_cuda=True)`). `compile.py` will fall back to CPU if no
  GPU is available.
- `trt_build.py` and the `trt_*` backends in `inference.py` additionally need the
  `tensorrt` and `onnx` packages installed and importable.
- The dataset is read from a fixed absolute path set in [utils.py](utils.py):
  `DATA_ROOT = "/mnt/ac142464/data/datasets/ddacs"`. Make sure this mount is available,
  or edit `DATA_ROOT` if your environment differs.
- The model config is read from `configs/model/model.yaml` at the repo root.

## Recommended order

The scripts build on each other's output artifacts in `ModelArtifacts/`. Run them in
this order the first time:

1. **`model_save.py`** — produces `model_eager.pt` (TorchScript export of the eager checkpoint).
2. **`compile.py`** — produces `model_default.pt`, `model_reduced_overhead.pt`, `model_max_autotune.pt`.
3. **`trt_build.py`** — produces `model.onnx`, `model_fp32.trt`, `model_tf32.trt`.
4. **`latency_profiling.py`** — benchmarks whatever eager/scripted/compiled artifacts already exist.
5. **`inference.py`** — compares every artifact that exists (including ONNX/TensorRT); safely skips any that are missing.

Each script is otherwise independent — you only need to (re-)run the ones that
produce the artifacts you're interested in.

## Running each script

### `model_save.py` — export TorchScript

```bash
uv run python HelperScripts/model_save.py
```

Loads `Model_ContactAware`, restores weights from a Lightning checkpoint, scripts it
with `model.to_torchscript(method="script")`, and saves `model_eager.pt` into the
current working directory.

> **Note:** this script currently has the model config path and checkpoint path
> hardcoded at the top of the file (`/home/RUS_CIP/st189432/MT-3970/configs/model/model.yaml`
> and an `mlruns/.../checkpoints/...ckpt` path under a different local directory). Update
> those two paths to point at your own config and checkpoint before running, and run it
> from `HelperScripts/` (or move `model_eager.pt` into `ModelArtifacts/` afterwards) so
> the other scripts can find it at `ModelArtifacts/model_eager.pt`.

### `compile.py` — `torch.compile` modes + accuracy check

```bash
uv run python HelperScripts/compile.py
```

For each of the three `torch.compile` modes (`default`, `reduce-overhead`,
`max-autotune`), loads a fresh eager model, compiles it, runs a 10-sample no-grad
inference pass over the DDACS test split, prints MAE/MSE/RMSE, and saves the compiled
module to `ModelArtifacts/model_<name>.pt`. Requires `ModelArtifacts/eager.ckpt` to
exist (the eager checkpoint used by `load_eager_model`).

### `trt_build.py` — ONNX export + TensorRT engine build

```bash
uv run python HelperScripts/trt_build.py
```

Traces the eager model on one test sample, exports `ModelArtifacts/model.onnx`, then
builds two TensorRT engines from it — `model_fp32.trt` (TF32 disabled, strict FP32) and
`model_tf32.trt` (TF32 enabled). Requires a CUDA GPU and `ModelArtifacts/eager.ckpt`.

### `latency_profiling.py` — latency benchmarking

```bash
uv run python HelperScripts/latency_profiling.py
```

For every artifact found in `ModelArtifacts/` (`eager`, `eager_scripted`, `default`,
`reduced_overhead`, `max_autotune`), runs 20 untimed warm-up passes followed by 100
timed inferences (10 files × 10 runs) using CUDA events, prints mean/p50/p90 latency,
and saves a Chrome trace per model to `ModelArtifacts/traces/<name>/single_inference_trace.json`
(viewable at `chrome://tracing` or in the PyTorch Profiler viewer). Missing artifacts
are skipped with a message. Requires a CUDA GPU.

### `inference.py` — full backend comparison

```bash
# One random test-split simulation
uv run python HelperScripts/inference.py

# 5 random simulations, reproducible via a fixed seed
uv run python HelperScripts/inference.py --num-random 5 --seed 0

# Specific simulation IDs (from process_parameters.csv)
uv run python HelperScripts/inference.py --sim-ids 16039 16040
```

Runs a single forward pass per (model, sample) pair across every backend found in
`ModelArtifacts/` — `eager`, `eager_scripted`, `default`, `reduced_overhead`,
`max_autotune`, `onnx`, `trt_fp32`, `trt_tf32` — using identical inputs for each.
Predictions and ground truth are denormalized back to physical mm, MAE/MSE/RMSE are
printed in a summary table, and a 3-panel plot (ground truth | prediction | signed
error surface) is saved per `(sim_id, model)` to `PLOTS/<sim_id>_<model_key>.png` at
the repo root. Missing artifacts or backends that fail to load are skipped rather than
aborting the run. Requires a CUDA GPU.

Flags:
- `--sim-ids ID [ID ...]` — run specific simulation IDs from the test split.
- `--num-random N` (default `1`) — number of random test-split samples, used when `--sim-ids` is not given.
- `--seed N` — seed for `--num-random` sampling, for reproducibility.

## Output locations

- `ModelArtifacts/` — all exported/compiled models, ONNX graph, TensorRT engines, and profiler traces.
- `PLOTS/` — ground-truth vs. prediction surface plots from `inference.py`.
