"""Utilities for reproducing MLflow runs.

Provides functions to load configs from past runs and re-run experiments.

Usage:
    uv run python -m src.utils.reproduce <run_id>
    uv run python -m src.utils.reproduce <run_id> --info
    uv run python -m src.utils.reproduce <run_id> --dry-run
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def get_run_info(run_id: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
    """Get details about an MLflow run.

    Args:
        run_id: MLflow run ID.
        tracking_uri: MLflow tracking URI. If None, uses default.

    Returns:
        Dictionary with run information including status, metrics, params, tags.

    Raises:
        ValueError: If run not found.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        raise ValueError(f"Run '{run_id}' not found: {e}")

    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def load_config_from_run(run_id: str, tracking_uri: Optional[str] = None) -> OmegaConf:
    """Load config from an MLflow run.

    Args:
        run_id: MLflow run ID.
        tracking_uri: MLflow tracking URI. If None, uses default.

    Returns:
        OmegaConf DictConfig with the run's configuration.

    Raises:
        FileNotFoundError: If config artifact not found in run.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            config_path = client.download_artifacts(run_id, "config.yaml", tmpdir)
        except Exception:
            try:
                config_path = client.download_artifacts(
                    run_id, "hydra_config.yaml", tmpdir
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"No config artifact found in run '{run_id}'. "
                    f"Expected 'config.yaml' or 'hydra_config.yaml'. Error: {e}"
                )

        cfg = OmegaConf.load(config_path)

    return cfg


def reproduce_run(
    run_id: str,
    tracking_uri: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[subprocess.CompletedProcess]:
    """Re-run experiment from stored config.

    Loads the config from the specified run and executes main.py
    with the same parameters. Adds reproduction metadata to distinguish
    from original runs.

    Args:
        run_id: MLflow run ID to reproduce.
        tracking_uri: MLflow tracking URI. If None, uses default.
        dry_run: If True, print command without executing.

    Returns:
        CompletedProcess if executed, None if dry_run.
    """
    cfg = load_config_from_run(run_id, tracking_uri)

    # Add reproduction metadata to config
    if "reproduction" not in cfg:
        OmegaConf.update(cfg, "reproduction", {})
    OmegaConf.update(cfg, "reproduction.reproduced_from", run_id)
    OmegaConf.update(cfg, "reproduction.is_reproduction", True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(cfg, f)
        config_path = f.name

    cmd = [
        sys.executable,
        "main.py",
        "--config-path",
        str(Path(config_path).parent),
        "--config-name",
        Path(config_path).stem,
    ]

    log.info(f"Reproducing run: {run_id}")
    log.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        log.info("(dry run - not executing)")
        return None

    result = subprocess.run(cmd, check=False)
    return result


def _setup_cli_logging() -> None:
    """Configure logging for CLI usage with stdout output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger(__name__).addHandler(handler)
    logging.getLogger(__name__).setLevel(logging.INFO)


def main() -> None:
    """CLI entry point for reproduce utility."""
    _setup_cli_logging()

    parser = argparse.ArgumentParser(
        description="Reproduce an MLflow run from stored config"
    )
    parser.add_argument("run_id", type=str, help="MLflow run ID to reproduce")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///./mlruns/mlflow.db",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show run info without reproducing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )

    args = parser.parse_args()

    if args.info:
        info = get_run_info(args.run_id, args.tracking_uri)
        log.info(f"\nRun: {info['run_id']}")
        log.info(f"Status: {info['status']}")
        log.info(f"Artifact URI: {info['artifact_uri']}")
        log.info("\nMetrics:")
        for k, v in sorted(info["metrics"].items()):
            log.info(f"  {k}: {v}")
        log.info("\nTags:")
        for k, v in sorted(info["tags"].items()):
            if not k.startswith("mlflow."):
                log.info(f"  {k}: {v}")
    else:
        reproduce_run(args.run_id, args.tracking_uri, args.dry_run)


if __name__ == "__main__":
    main()
