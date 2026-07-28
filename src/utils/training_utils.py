"""Training utilities for PyTorch Lightning.

Provides helper functions for setting up callbacks and metrics.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from src.metrics.metric_collection import MetricCollection

if TYPE_CHECKING:
    from pytorch_lightning.loggers import MLFlowLogger

    from src.utils.config_manager import ConfigManager

log = logging.getLogger(__name__)


def setup_metrics(metric_names: Optional[List[str]] = None) -> MetricCollection:
    """Setup evaluation metrics.

    Args:
        metric_names: Optional list of metric names to include.
                      If None, returns all available metrics.

    Returns:
        MetricCollection with selected (or all) metrics.

    Example:
        >>> metrics = setup_metrics()  # All metrics
        >>> metrics = setup_metrics(["mse", "rmse", "mae"])  # Only these
    """
    all_metrics = MetricCollection.create_default()

    if metric_names is not None:
        return all_metrics.select(metric_names)

    return all_metrics


def setup_callbacks(config: "ConfigManager", logger: "MLFlowLogger") -> List:
    """Setup training callbacks.

    Args:
        config: ConfigManager instance.
        logger: MLFlowLogger instance (used to get artifact path for checkpoints).

    Returns:
        List of PyTorch Lightning callbacks.
    """
    from pathlib import Path

    import mlflow

    callbacks = []

    # Save checkpoints directly to MLflow's artifact folder for automatic tracking
    run = mlflow.get_run(logger.run_id)
    artifact_uri = run.info.artifact_uri
    checkpoint_dir = Path(artifact_uri) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cfg = config.checkpoint_config
    if ckpt_cfg:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=ckpt_cfg.get("filename", "epoch={epoch:02d}"),
            monitor=ckpt_cfg.get("monitor", "val/loss"),
            mode=ckpt_cfg.get("mode", "min"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            save_last=ckpt_cfg.get("save_last", True),
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    es_cfg = config.early_stopping_config
    if es_cfg.get("enabled", False):
        early_stop_callback = EarlyStopping(
            monitor=es_cfg.get("monitor", "val/loss"),
            mode=es_cfg.get("mode", "min"),
            patience=es_cfg.get("patience", 15),
            min_delta=es_cfg.get("min_delta", 0.0001),
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Reshuffles streaming datasets each epoch; no-op for map-style datasets.
    callbacks.append(StreamingShuffleCallback())

    return callbacks


class StreamingShuffleCallback(Callback):
    """Call ``set_epoch`` on streaming train datasets at each epoch start.

    IterableDatasets like ``ddacs.pytorch.DDACSDataset`` shuffle per
    worker-shard with a seed derived from the epoch; without ``set_epoch``
    every epoch would iterate in the same order. Datasets without a
    ``set_epoch`` method are ignored, so this callback is always safe.
    """

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Propagate the current epoch to the train dataset if supported."""
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)
