"""Archive the final submission to the institute's model store.

The last step of a thesis submission: copies everything needed to reuse
your results - independent of your home directory and mlruns - to
``<models_root>/<thesis_id>/`` (default ``/mnt/data/models/<thesis_id>/``)::

    <thesis_id>/
    ├── MANIFEST.json    # run_id, model version, git commit, student, date
    ├── config.yaml      # exact training configuration
    ├── metrics.json     # all metrics + params of the registered run
    ├── model/           # the MLflow pytorch model (loadable anywhere)
    └── figures/         # output of scripts/make_figures.py

Usage (after `evaluate.py --register` and `scripts/make_figures.py`):
    uv run python -m src.utils.archive_submission
    uv run python -m src.utils.archive_submission --force   # overwrite
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

TRACKING_URI = "sqlite:///./mlruns/mlflow.db"


def _load_config() -> OmegaConf:
    """Load configs/config.yaml (without Hydra composition)."""
    return OmegaConf.load("configs/config.yaml")


def _thesis_id(cfg) -> str:
    """Assemble the thesis id (same scheme as ConfigManager)."""
    work_type = cfg.get("work_type")
    thesis_number = cfg.get("thesis_number")
    if not work_type or "CHANGE_ME" in str((work_type, thesis_number)):
        raise SystemExit(
            "Set work_type and thesis_number in configs/config.yaml first."
        )
    return f"{work_type}_{thesis_number}"


def _git_commit() -> str:
    """Current git commit hash (submission must be committed)."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return "unknown"
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
    return result.stdout.strip() + ("-dirty" if dirty else "")


def archive(force: bool = False) -> Path:
    """Copy the registered model, config, metrics and figures to models_root.

    Args:
        force: Overwrite an existing archive.

    Returns:
        Path of the created archive directory.
    """
    cfg = _load_config()
    thesis_id = _thesis_id(cfg)
    models_root = Path(OmegaConf.select(cfg, "paths.models_root"))

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{thesis_id}'")
    if not versions:
        raise SystemExit(
            f"No registered model '{thesis_id}'. Register your best run first:\n"
            f"  uv run evaluate.py --run_id <id> --datasets all --register"
        )
    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)

    target = models_root / thesis_id
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists. Rerun with --force to overwrite.")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    logger.info(f"archiving '{thesis_id}' (version {latest.version}) -> {target}")

    # model + config straight from the run's artifacts
    mlflow.artifacts.download_artifacts(
        run_id=latest.run_id, artifact_path="model", dst_path=str(target)
    )
    config_path = mlflow.artifacts.download_artifacts(
        run_id=latest.run_id, artifact_path="config.yaml", dst_path=str(target)
    )
    logger.info(f"model + {Path(config_path).name} copied")

    (target / "metrics.json").write_text(
        json.dumps({"metrics": run.data.metrics, "params": run.data.params}, indent=2)
    )

    figures = Path("figures")
    if figures.is_dir() and any(figures.iterdir()):
        shutil.copytree(figures, target / "figures")
        logger.info(f"figures/ copied ({sum(1 for _ in figures.iterdir())} files)")
    else:
        logger.warning("no figures/ directory - run scripts/make_figures.py first")

    manifest = {
        "thesis_id": thesis_id,
        "student_name": cfg.get("student_name"),
        "run_id": latest.run_id,
        "model_version": int(latest.version),
        "git_commit": _git_commit(),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "load_with": f"mlflow.pytorch.load_model('{target}/model')",
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    logger.info("archive complete")
    return target


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing archive"
    )
    args = parser.parse_args()
    target = archive(force=args.force)
    print(f"\nSubmission archived: {target}")
    sys.exit(0)


if __name__ == "__main__":
    main()
