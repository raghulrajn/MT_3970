# Configuration

This project uses [Hydra](https://hydra.cc/docs/intro/) for config composition and `ConfigManager` for convenient access to paths, datasets, and MLflow settings.

## Data Location

Datasets are on shared storage at `/mnt/data/` (or set `DATA_DIR` environment variable):

```bash
export DATA_DIR="/mnt/data"  # add to ~/.bashrc
```

## Basic Usage

```python
from src.utils.config_manager import ConfigManager

@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    config = ConfigManager.from_hydra(cfg)

    dataset_path = config.get_dataset_path()
    tracking_uri = config.get_mlflow_tracking_uri()
    lr = config.get("model.optimizer.learning_rate", default=0.001)
```

## Quick Reference

| Method / Property | Returns |
|---|---|
| `config.get_dataset_path()` | Path to current dataset |
| `config.list_available_datasets()` | `['ddacs/h5', 'ddacs/pointcloud', ...]` |
| `config.get_mlflow_tracking_uri()` | `sqlite:///./mlruns/mlflow.db` |
| `config.mlflow_tags` | Dict with model, dataset, thesis info |
| `config.get("key.nested", default)` | Config value by dot notation |
| `config.model_config` | Model sub-config |
| `config.dataset_config` | Dataset sub-config |
| `config.training_config` | Training sub-config |
| `config.thesis_id` | e.g. `MT_2024_042` |
| `config.submission_id` | e.g. `MT_2024_042_dummy_ddacs` |
| `config.validate()` | Validate config setup |

## Loading from YAML

For standalone scripts (e.g. `evaluate.py`):

```python
config = ConfigManager.from_yaml("path/to/config.yaml")
```

See the full API in the [ConfigManager reference](../api/utils).
