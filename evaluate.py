"""Model evaluation script for testing trained models from MLflow.

Loads a model from an MLflow run and evaluates it on specified datasets.
Can optionally register the model to the Model Registry after evaluation.

Usage:
    # Evaluate a run on specific datasets
    uv run evaluate.py --run_id abc123 --datasets ddacs hsh

    # Evaluate and register the model
    uv run evaluate.py --run_id abc123 --datasets ddacs --register
"""

import argparse
import logging
from typing import Any, Dict, List

import mlflow
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import get_datamodule
from src.utils.config_manager import ConfigManager
from src.utils.file_utils import list_dataset_configs
from src.utils.mlflow_utils import (
    get_tracking_uri,
    load_model_and_config,
    register_model,
)
from src.utils.training_utils import setup_metrics

log = logging.getLogger(__name__)


def _load_base_config() -> dict:
    """Load base config.yaml file."""
    import yaml

    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> None:
    """Configure logging from config file."""
    cfg = _load_base_config()
    log_format = (
        cfg.get("hydra", {})
        .get("job_logging", {})
        .get("formatters", {})
        .get("simple", {})
        .get("format", "%(asctime)s [%(levelname)s] - %(message)s")
    )
    logging.basicConfig(level=logging.INFO, format=log_format)


def evaluate_on_dataset(
    model: pl.LightningModule,
    dataset_name: str,
    config: ConfigManager,
) -> Dict[str, Any]:
    """Evaluate model on a specific dataset.

    Args:
        model: PyTorch Lightning model.
        dataset_name: Dataset name (e.g., 'ddacs').
        config: ConfigManager with modality info.

    Returns:
        Dictionary with dataset name and metrics.
    """
    log.info(f"Evaluating on {dataset_name}...")

    dataset_cfg = config.dataset_config.copy()

    if dataset_name != dataset_cfg.get("name"):
        new_dataset_cfg = OmegaConf.load(f"configs/dataset/{dataset_name}.yaml")

        datasets_root = config.get("paths.datasets_root", "/mnt/data/datasets")
        resolver_cfg = OmegaConf.create({"paths": {"datasets_root": datasets_root}})
        new_dataset_cfg = OmegaConf.merge(resolver_cfg, new_dataset_cfg)
        new_dataset_cfg = OmegaConf.to_container(new_dataset_cfg, resolve=True)

        dataset_cfg.update(
            {
                "name": new_dataset_cfg.get("name", dataset_name),
                "modalities": new_dataset_cfg.get(
                    "modalities", dataset_cfg.get("modalities")
                ),
            }
        )

    # Build config for get_datamodule
    cfg = OmegaConf.create(
        {
            "dataset": dataset_cfg,
            "training": config.training_config,
        }
    )

    datamodule = get_datamodule(cfg)

    # Setup metrics from config
    metric_names = config.model_config.get("metrics")
    metrics = setup_metrics(metric_names)
    model.setup_metrics(metrics)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    results = trainer.test(model, datamodule=datamodule)

    if not results:
        raise RuntimeError(f"Evaluation failed on {dataset_name}")

    return {
        "dataset": dataset_name,
        "metrics": results[0],
    }


def main() -> None:
    """Main evaluation function."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate model from MLflow run on datasets"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="MLflow run ID to load model from",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to evaluate on (e.g., ddacs hsh) or 'all'",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model to Model Registry after evaluation",
    )

    args = parser.parse_args()

    tracking_uri = get_tracking_uri()
    model, config, run = load_model_and_config(args.run_id, tracking_uri)

    model_name = config.model_config.get("name")
    modality = config.dataset_config.get("modality")
    log.info(f"Model: {model_name}")
    log.info(f"Modality: {modality}")

    if args.datasets == ["all"]:
        datasets = list_dataset_configs()
    else:
        datasets = args.datasets

    log.info(f"Datasets: {datasets}")

    all_results: List[Dict[str, Any]] = []

    for dataset in datasets:
        try:
            result = evaluate_on_dataset(model, dataset, config)
            all_results.append(result)
            log.info(f"  {dataset}: Done")
        except Exception as e:
            log.error(f"  {dataset}: Failed - {e}")

    if all_results:
        summary_data = []
        for result in all_results:
            row = {"dataset": result["dataset"]}
            row.update(result["metrics"])
            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        log.info(f"\nResults:\n{df.to_string(index=False)}")

        # Log evaluation metrics back to the run
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=args.run_id):
            for result in all_results:
                dataset = result["dataset"]
                for metric_name, value in result["metrics"].items():
                    mlflow.log_metric(f"eval/{dataset}/{metric_name}", value)

        if args.register:
            avg_metrics = {}
            for col in df.columns:
                if col != "dataset":
                    avg_metrics[col] = df[col].mean()

            # Use thesis_id for registered model name (e.g., MT_7890)
            # This creates versioned milestones: MT_7890/1, MT_7890/2, etc.
            registered_name = run.data.tags.get("thesis_id", f"{model_name}_model")
            model_uri = register_model(
                args.run_id, registered_name, avg_metrics, datasets, tracking_uri
            )
            log.info(f"Registered: {model_uri}")
            log.info(f"Load with: mlflow.pytorch.load_model('{model_uri}')")


if __name__ == "__main__":
    main()
