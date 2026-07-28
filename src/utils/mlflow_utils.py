"""MLflow utilities for logging datasets, parameters, and configs.

Provides helper functions for MLflow integration that focus on:
- Dataset metadata logging (NOT actual data, just pointers)
- Key parameter logging for filtering and comparison
- Full config logging for reproduction
- Logger setup for PyTorch Lightning
"""

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import mlflow
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger, MLFlowLogger

from src.utils.config_manager import ConfigManager

# MLflow configuration
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_LOCK_MODEL_DEPENDENCIES"] = "true"


if TYPE_CHECKING:
    from src.visualization.pointcloud import plot_pointclouds

# Suppress verbose logging
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
logging.getLogger("alembic.runtime.plugins").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

# Suppress SLURM srun hint
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python3 main.py",
)


log = logging.getLogger(__name__)


class LocalPathSource(
    mlflow.data.dataset_source.DatasetSource
):  # pyright: ignore[reportPrivateImportUsage, reportAttributeAccessIssue]
    """Dataset source for local filesystem paths."""

    def __init__(self, path: str):
        self._path = path

    @staticmethod
    def _get_source_type() -> str:
        return "local"

    def load(self, dst_path=None):
        """Loads the path of the dataset."""
        return self._path

    @staticmethod
    def _can_resolve(raw_source):
        return False

    @staticmethod
    def _resolve(raw_source):
        return None

    def to_dict(self):
        """Transforms itself into a dictionary."""
        return {"path": self._path}

    @classmethod
    def from_dict(cls, source_dict):
        """Creates frm dict."""
        return cls(source_dict["path"])


def log_datamodule(
    name: str,
    path: str,
    modality: str,
    metadata: Optional[Dict[str, Any]] = None,
    context: str = "training",
) -> None:
    """Log DataModule metadata as MLflow Dataset entity.

    This logs metadata only, NOT actual data. The data is too big,
    so we just store a pointer to the path.

    Args:
        name: Dataset identifier (e.g., 'ddacs').
        path: Path where data lives on disk.
        modality: Type of data representation (e.g., 'pointcloud', 'graph').
        metadata: Optional dict with description, num_samples, etc.
        context: Context for the dataset (e.g., 'training', 'validation').

    Example:
        log_datamodule(
            name="ddacs",
            path="/mnt/data/datasets/ddacs/pointcloud",
            modality="pointcloud",
            metadata={"description": "DDACS dataset", "num_samples": 32071}
        )
    """
    dataset_name = f"{name}/{modality}"

    source = LocalPathSource(path)
    dataset = mlflow.data.meta_dataset.MetaDataset(  # pyright: ignore[reportPrivateImportUsage]
        source=source,
        name=dataset_name,
        digest=f"{name}_{modality}",
    )

    tags = {}
    if metadata:
        if "description" in metadata:
            tags["description"] = str(metadata["description"])
        if "num_samples" in metadata:
            tags["num_samples"] = str(metadata["num_samples"])
        if "doi" in metadata:
            tags["doi"] = str(metadata["doi"])

    mlflow.log_input(dataset, context=context, tags=tags)


def log_key_params(cfg: DictConfig, keys: List[str]) -> None:
    """Log only specified parameters from config.

    Use this to log important hyperparameters for filtering and comparison
    in the MLflow UI, without cluttering with all 100+ config values.

    Args:
        cfg: Hydra DictConfig object.
        keys: List of dot-notation keys to log (e.g., ['model.optimizer.learning_rate']).

    Example:
        log_key_params(cfg, [
            "model.name",
            "model.optimizer.learning_rate",
            "model.architecture.hidden_dims",
            "training.batch_size",
            "training.max_epochs",
        ])
    """
    params = {}
    for key in keys:
        value = OmegaConf.select(cfg, key, default=None)
        if value is not None:
            flat_key = key.replace(".", "_")
            params[flat_key] = value

    if params:
        mlflow.log_params(params)


def log_full_config(cfg: DictConfig, filename: str = "config.yaml") -> None:
    """Log full config as artifact for reproduction.

    Saves the complete resolved Hydra config as a YAML file artifact,
    allowing exact reproduction of any run.

    Args:
        cfg: Hydra DictConfig object.
        filename: Name of the artifact file.

    Example:
        log_full_config(cfg)
        # Later: load with mlflow.artifacts.download_artifacts()
    """
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    mlflow.log_text(yaml_str, filename)


def setup_file_logging(prefix: str = "training") -> str:
    """Setup file handler to capture logs for MLflow artifact.

    Uses Hydra's runtime output directory.

    Args:
        prefix: Prefix for the log file name.

    Returns:
        Path to the log file.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log_file_name = str(output_dir / f"{prefix}_output.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    return log_file_name


def get_key_params_list() -> List[str]:
    """Get default list of key parameters to track.

    Returns:
        List of important parameter keys for MLflow logging.
    """
    return [
        "model.name",
        "model.optimizer.learning_rate",
        "model.optimizer.weight_decay",
        "model.architecture.hidden_dims",
        "model.architecture.dropout",
        "model.scheduler.enabled",
        "dataset.name",
        "dataset.modality",
        "training.batch_size",
        "training.max_epochs",
        "training.gradient_clip_val",
        "seed",
    ]


def log_sample_visualization(
    datamodule,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Create and optionally save a sample visualization from the datamodule.

    Args:
        datamodule: PyTorch Lightning DataModule with setup() called.
        save_path: Path to save the figure. If None, creates a temp file.

    Returns:
        Path to the saved figure, or None if visualization failed.
    """
    try:

        datamodule.setup("fit")
        dataset = datamodule.train_dataset

        if len(dataset) == 0:
            log.warning("Dataset is empty, skipping visualization")
            return None

        sample = dataset[0]

        if save_path is None:
            save_path = "sample_input_data.png"

        plot_pointclouds(
            sample, title="Training Sample", save_path=save_path, show=False
        )
        return save_path

    except Exception as e:
        log.warning(f"Could not create sample visualization: {e}")
        return None


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from config file.

    Reads the base config.yaml without Hydra to get the tracking URI.
    Use this when running outside of Hydra context.

    Returns:
        MLflow tracking URI string.
    """
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    mlflow_dir = cfg.get("mlflow", {}).get("tracking_uri", "./mlruns")
    return f"sqlite:///{mlflow_dir}/mlflow.db"


def load_model_and_config(run_id: str, tracking_uri: str) -> tuple:
    """Load model and config from MLflow run.

    Args:
        run_id: MLflow run ID.
        tracking_uri: MLflow tracking URI.

    Returns:
        Tuple of (model, config_manager, run_info).

    Raises:
        RuntimeError: If run, model artifact, or config cannot be found.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        log.error(f"Run not found: {run_id}")
        log.error("Verify the run_id is correct and the tracking URI is set properly.")
        log.error(f"Tracking URI: {tracking_uri}")
        raise RuntimeError(f"Failed to find run '{run_id}': {e}") from e

    log.info(f"Loading from run: {run_id}")

    model_name = run.data.params.get("model_name", run.data.tags.get("model_name"))
    if not model_name:
        log.error("No model_name found in run params or tags.")
        log.error(f"Available params: {list(run.data.params.keys())}")
        log.error(f"Available tags: {list(run.data.tags.keys())}")
        raise RuntimeError(f"Cannot determine model name from run '{run_id}'")

    model_uri = f"runs:/{run_id}/{model_name}"
    log.info(f"Loading model from: {model_uri}")
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
    except Exception as e:
        log.error(f"Failed to load model from: {model_uri}")
        log.error(f"Error: {e}")
        log.error("The model may not have been logged with mlflow.pytorch.log_model().")
        log.error("Re-run training or check main.py model logging code.")
        raise RuntimeError(f"Failed to load model: {e}") from e

    try:
        config_path = client.download_artifacts(run_id, "config.yaml")
        config = ConfigManager.from_yaml(config_path)
        log.info(
            f"Loaded config with modality: {config.dataset_config.get('modality')}"
        )
    except Exception as e:
        log.error("Failed to load config from run artifacts.")
        log.error(f"Error: {e}")
        log.error("The config.yaml may not have been logged as an artifact.")
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        log.error(f"Available artifacts: {artifact_names}")
        raise RuntimeError(f"Failed to load config: {e}") from e

    return model, config, run


def register_model(
    run_id: str,
    model_name: str,
    metrics: Dict[str, float],
    datasets: List[str],
    tracking_uri: str,
) -> str:
    """Register model to MLflow Model Registry.

    Each registration creates a new version, allowing milestone tracking.

    Args:
        run_id: MLflow run ID.
        model_name: Name for the registered model (thesis_id).
        metrics: Evaluation metrics to add as description.
        datasets: List of datasets evaluated on.
        tracking_uri: MLflow tracking URI.

    Returns:
        Registered model URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    artifact_model_name = run.data.params.get(
        "model_name", run.data.tags.get("model_name")
    )
    model_uri = f"runs:/{run_id}/{artifact_model_name}"

    metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    description = (
        f"Model: {artifact_model_name} | "
        f"Datasets: {', '.join(datasets)} | "
        f"Metrics: {metrics_str}"
    )

    result = mlflow.register_model(model_uri, model_name)
    client.update_model_version(
        name=model_name,
        version=result.version,
        description=description,
    )

    log.info(f"Registered model: {model_name} version {result.version}")
    return f"models:/{model_name}/{result.version}"


def setup_logger(config: "ConfigManager") -> Logger:
    """Setup MLflow experiment logger.

    Args:
        config: ConfigManager instance.

    Returns:
        PyTorch Lightning MLFlowLogger instance.
    """
    log_cfg = config.logging_config

    experiment_name = log_cfg.get("experiment_name", config.submission_id)
    tracking_uri = config.get_mlflow_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)

    logger = MLFlowLogger(
        save_dir=str(config.mlflow_dir),
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=config.run_name,
        tags=config.mlflow_tags,
        log_model=False,  # We log the model manually with mlflow.pytorch.log_model()
    )

    log.info(
        f'View experiments ("{experiment_name}"): '
        f"uv run mlflow ui --backend-store-uri {tracking_uri}"
    )

    return logger
