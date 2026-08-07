"""Main script for training and evaluating surrogate models.

Usage:
    # Train with default config
    uv run main.py

    # Override config parameters
    uv run main.py model=your_model dataset=your_dataset training.max_epochs=200

    # Run hyperparameter sweep
    uv run main.py -m model.optimizer.lr=0.001,0.0001 model.architecture.hidden_dims=[64,32],[128,64]
"""

import logging
import os

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.data import get_datamodule
from src.utils.config_manager import ConfigManager
from src.utils.mlflow_utils import (
    get_key_params_list,
    log_datamodule,
    log_full_config,
    log_key_params,
    setup_file_logging,
    setup_logger,
)
from src.utils.model_registry import get_model
from src.utils.training_utils import setup_callbacks, setup_metrics

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Sets up experiment directories, loads data and model, configures trainer,
    and runs training and evaluation pipeline.

    Args:
        cfg: Hydra configuration with model, data, and training parameters.
    """
    log_file = setup_file_logging()

    config = ConfigManager.from_hydra(cfg)
    log.info(f"Thesis ID: {config.thesis_id}")

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True, verbose=False)
        log.info(f"Seed set to: {cfg.seed}")

    if cfg.training.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        log.info("Deterministic mode enabled (may reduce speed by ~10-15%)")

    log.info("Setting up Logger...")
    logger = setup_logger(config)

    log.info("Setting up data module...")
    datamodule = get_datamodule(cfg)

    with mlflow.start_run(run_id=logger.run_id):
        log_key_params(cfg, get_key_params_list())
        log_full_config(cfg)

        modality_cfg = cfg.dataset.modalities[cfg.dataset.modality]
        log_datamodule(
            name=cfg.dataset.name,
            path=modality_cfg.data_dir,
            modality=cfg.dataset.modality,
            metadata=dict(cfg.dataset.get("metadata", {})),
        )
        log.warning(
            "Sample visualization is not yet implemented for the "
            "current pointcloud format. Skipping."
        )

    log.info(f"Setting up model: {cfg.model.name}")
    model = get_model(cfg)

    log.info("Setting up evaluation metrics...")
    metric_names = cfg.model.get("metrics", None)
    metrics = setup_metrics(metric_names)
    log.info(f"Tracking metrics: {metrics.available_metrics()}")
    model.setup_metrics(metrics)

    log.info("Setting up callbacks...")
    callbacks = setup_callbacks(config, logger)

    log.info("Setting up trainer...")
    # data_fraction sets a uniform train/val/test limit (e.g. 0.1 = 10% of each
    # split, every epoch). Explicit limit_*_batches overrides (e.g. fast_debug's
    # fixed pipeline-check batch counts) always take precedence over it.
    data_fraction = cfg.training.get("data_fraction", 1.0)
    total_samples = cfg.dataset.get("metadata", {}).get("num_samples")
    if total_samples:
        log.info(
            f"Using {data_fraction:.0%} of the data "
            f"(~{int(total_samples * data_fraction)} of {total_samples} simulations, "
            f"split across train/val/test)."
        )
    else:
        log.info(f"Using {data_fraction:.0%} of the data (per split).")
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.benchmark,
        limit_train_batches=cfg.training.get("limit_train_batches", data_fraction),
        limit_val_batches=cfg.training.get("limit_val_batches", data_fraction),
        limit_test_batches=cfg.training.get("limit_test_batches", data_fraction),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        check_val_every_n_epoch=1,
    )

    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Load best checkpoint for logging and testing
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        log.info(f"Best checkpoint: {best_ckpt}")
        best_model = model.__class__.load_from_checkpoint(best_ckpt, weights_only=False)
        best_model.setup_metrics(metrics)  # Re-setup metrics on loaded model
    else:
        log.warning("No best checkpoint found, using final model state")
        best_model = model

    with mlflow.start_run(run_id=logger.run_id):
        model_name = cfg.model.name
        register_model = cfg.mlflow.get("register_model", False)
        mlflow.log_param("model_name", model_name)
        if best_ckpt:
            mlflow.log_param("best_checkpoint", best_ckpt)

        try:
            mlflow.pytorch.log_model(
                pytorch_model=best_model,
                name=model_name,
                registered_model_name=model_name if register_model else None,
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
            )
            log.info(
                f"Logged model artifact '{model_name}' (registered={register_model})"
            )
        except Exception as e:
            log.error(f"Failed to log model artifact: {e}")
            raise

    log.info("Starting testing...")
    trainer.test(model=best_model, datamodule=datamodule)

    log.info(f"Training complete! Run ID: {logger.run_id}")

    run = mlflow.get_run(logger.run_id)
    log.info(f"Artifact stored here: {run.info.artifact_uri}")

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_artifact(log_file, artifact_path="logs")


if __name__ == "__main__":
    main()
