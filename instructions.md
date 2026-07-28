# ML Template Refactoring: Hydra + MLflow Integration

## Current Status: COMPLETE

The refactoring is complete. This document tracks the work done.

## Completed Work

### Architecture

- **Hydra**: Handles config composition only. Outputs go to `/tmp/hydra/` and are ignored.
- **MLflow**: Single source of truth for all experiment data stored in `./mlruns/`

### Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `main.py` | Training script with MLflow logging | ✅ Complete |
| `evaluate.py` | Evaluate model on datasets, register to Model Registry | ✅ Complete |
| `src/utils/reproduce.py` | Reproduce runs from stored config | ✅ Complete |
| `src/utils/mlflow_utils.py` | MLflow logging utilities | ✅ Complete |
| `tests/test_submission.py` | Submission validation tests | ✅ Complete |

### Key Features Implemented

1. **MLflow Model Registry workflow**:
   - Register models as `models:/MT_7890/1`, `models:/MT_7890/2`, etc.
   - Each version is a milestone linked to source run
   - Load with `mlflow.pytorch.load_model('models:/MT_7890/latest')`

2. **Reproduction with metadata**:
   - Reproduced runs tagged with `is_reproduction: true` and `reproduced_from: <run_id>`
   - Run name prefixed with `[REPRODUCE]`

3. **Submission tests**:
   - `pytest tests/test_submission.py -v` validates registered model
   - Checks model exists, can be loaded, source run has config
   - Full reproduction test with metric comparison (20% tolerance)

4. **All artifacts in MLflow**:
   - Checkpoints saved to `mlruns/<exp>/<run>/artifacts/checkpoints/`
   - Config saved as artifact for reproduction
   - Model logged with `mlflow.pytorch.log_model()`

### Student Workflow

```bash
# 1. Train
uv run main.py dataset=hsh

# 2. View experiments
uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db

# 3. Evaluate and register best model
uv run evaluate.py --run_id <best_run_id> --datasets all --register

# 4. Validate submission
uv run pytest tests/test_submission.py -v

# 5. Reproduce a run (if needed)
uv run python -m src.utils.reproduce <run_id>
```

### Files Removed

- `submit.py` - Replaced by `evaluate.py --register`
- `best_run_dir` config option - Now uses Model Registry

### Code Cleanup Completed

The following unused methods were removed from `src/utils/config_manager.py`:
- `get_model_dir()` - centralized model storage (replaced by MLflow)
- `my_model_dir` property - wrapper for get_model_dir
- `get_checkpoint_path()` - centralized checkpoint storage (replaced by MLflow artifacts)
- `export_dir` property - centralized export directory (replaced by Model Registry)

The following unused function was removed from `src/utils/file_utils.py`:
- `find_single_checkpoint()` - not used anywhere

### Documentation Updated

All documentation files have been updated for the MLflow-centric workflow:
- `docs/EXPERIMENTS.md` - Complete rewrite with new workflow
- `docs/README.md` - Updated scripts section
- `docs/STUDENT_RULES.md` - Updated evaluation workflow
- `docs/CONFIG_MANAGER_USAGE.md` - Major rewrite for MLflow approach

## Future Improvements (Optional)

Consider adding:
- Helper command to find latest run_id
- Helper command to list registered models
- Better error messages when model not registered
- Additional unit tests for `src/utils/mlflow_utils.py` and `evaluate.py`

## Quick Reference

### Commands

```bash
# Training
uv run main.py                              # Default config
uv run main.py dataset=hsh                     # Different dataset
uv run main.py model.optimizer.learning_rate=1e-4      # Override params
uv run main.py -m model.optimizer.learning_rate=1e-3,1e-4  # Sweep

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db

# Evaluation
uv run evaluate.py --run_id <id> --datasets all
uv run evaluate.py --run_id <id> --datasets all --register

# Reproduction
uv run python -m src.utils.reproduce <run_id> --info
uv run python -m src.utils.reproduce <run_id> --dry-run
uv run python -m src.utils.reproduce <run_id>

# Testing
uv run pytest tests/test_submission.py -v -m "not slow"
uv run pytest tests/test_submission.py -v  # Full test with reproduction
```

### Load Registered Model

```python
import mlflow.pytorch
mlflow.set_tracking_uri("sqlite:///./mlruns/mlflow.db")
model = mlflow.pytorch.load_model("models:/MT_7890/latest")
```

### Get Run ID from Registered Model

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///./mlruns/mlflow.db")
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='MT_7890'")
latest = max(versions, key=lambda v: int(v.version))
print(f"Run ID: {latest.run_id}")
```

## Storage Structure

```
./mlruns/
├── mlflow.db                    # SQLite database
└── <experiment_id>/
    └── <run_id>/
        └── artifacts/
            ├── config.yaml      # Full config for reproduction
            ├── checkpoints/     # Best model checkpoint
            ├── logs/            # Training log
            └── <model_name>/    # Logged PyTorch model
```
