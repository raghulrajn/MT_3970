"""
Submission validation tests.

These tests verify that:
1. A model is registered in MLflow Model Registry
2. The model can be loaded and run inference on test data
3. The source run has correct metadata (thesis_id, config)
4. (Optional) The model can be fully reproduced with similar metrics

Run with:
    # Quick validation (recommended)
    pytest tests/test_submission.py -v -m "not slow"

    # Full validation including reproduction (can take hours!)
    pytest tests/test_submission.py -v

Note: These tests require that you have registered a model using:
    uv run evaluate.py --run_id <id> --datasets all --register
"""

import subprocess
import sys
from typing import Dict, Optional

import mlflow
import pytest
from mlflow.tracking import MlflowClient


def get_tracking_uri() -> str:
    """Get MLflow tracking URI from config."""
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    mlflow_dir = cfg.get("mlflow", {}).get("tracking_uri", "./mlruns")
    return f"sqlite:///{mlflow_dir}/mlflow.db"


def get_thesis_id() -> str:
    """Get thesis_id from config."""
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    work_type = cfg.get("work_type", "MT")
    thesis_number = cfg.get("thesis_number", "0000")
    return f"{work_type}_{thesis_number}"


@pytest.fixture(scope="module")
def mlflow_client() -> MlflowClient:
    """Setup MLflow client."""
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


@pytest.fixture(scope="module")
def thesis_id() -> str:
    """Get thesis ID for model registry."""
    return get_thesis_id()


class TestModelRegistration:
    """Test that a model is properly registered."""

    def test_registered_model_exists(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> None:
        """Test that models:/thesis_id/latest exists."""
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            versions = []

        assert len(versions) > 0, (
            f"No registered model found with name '{thesis_id}'.\n"
            f"Please register your best model using:\n"
            f"  uv run evaluate.py --run_id <your_best_run_id> --datasets all --register"
        )

    def test_latest_version_has_run_id(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> None:
        """Test that the latest version links to a source run."""
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            pytest.skip(f"Model '{thesis_id}' not registered")

        if not versions:
            pytest.skip(f"Model '{thesis_id}' not registered")

        # Get latest version
        latest = max(versions, key=lambda v: int(v.version))

        assert (
            latest.run_id is not None
        ), f"Model version {latest.version} has no source run_id"

    def test_source_run_exists(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> None:
        """Test that the source run exists and has required artifacts."""
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            pytest.skip(f"Model '{thesis_id}' not registered")

        if not versions:
            pytest.skip(f"Model '{thesis_id}' not registered")

        latest = max(versions, key=lambda v: int(v.version))
        run_id = latest.run_id

        # Check run exists
        try:
            _ = mlflow_client.get_run(run_id)
        except Exception as e:
            pytest.fail(f"Source run '{run_id}' not found: {e}")

        # Check config artifact exists
        artifacts = mlflow_client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]

        assert "config.yaml" in artifact_names, (
            f"config.yaml not found in run artifacts. " f"Available: {artifact_names}"
        )

    def test_model_can_be_loaded(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> None:
        """Test that the model can be loaded from registry."""
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            pytest.skip(f"Model '{thesis_id}' not registered")

        if not versions:
            pytest.skip(f"Model '{thesis_id}' not registered")

        model_uri = f"models:/{thesis_id}/latest"

        try:
            model = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            pytest.fail(f"Failed to load model from '{model_uri}': {e}")

        assert model is not None, "Model loaded as None"


class TestReproducibility:
    """Test that registered model can be reproduced."""

    def _get_source_run_id(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> Optional[str]:
        """Get the source run_id from the latest registered model version."""
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            return None

        if not versions:
            return None

        latest = max(versions, key=lambda v: int(v.version))
        return latest.run_id

    def _get_run_metrics(
        self, mlflow_client: MlflowClient, run_id: str
    ) -> Dict[str, float]:
        """Get test metrics from a run."""
        run = mlflow_client.get_run(run_id)
        return {k: v for k, v in run.data.metrics.items() if k.startswith("test/")}

    def test_reproduce_info(self, mlflow_client: MlflowClient, thesis_id: str) -> None:
        """Test that reproduce --info works for the source run."""
        run_id = self._get_source_run_id(mlflow_client, thesis_id)
        if not run_id:
            pytest.skip(f"Model '{thesis_id}' not registered")

        tracking_uri = get_tracking_uri()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.utils.reproduce",
                run_id,
                "--tracking-uri",
                tracking_uri,
                "--info",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"reproduce --info failed:\n{result.stderr}"
        assert (
            f"Run: {run_id}" in result.stdout
        ), f"Run ID not found in output:\n{result.stdout}"

    def test_reproduce_dry_run(
        self, mlflow_client: MlflowClient, thesis_id: str
    ) -> None:
        """Test that reproduce --dry-run generates correct command."""
        run_id = self._get_source_run_id(mlflow_client, thesis_id)
        if not run_id:
            pytest.skip(f"Model '{thesis_id}' not registered")

        tracking_uri = get_tracking_uri()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.utils.reproduce",
                run_id,
                "--tracking-uri",
                tracking_uri,
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"reproduce --dry-run failed:\n{result.stderr}"
        assert (
            "main.py" in result.stdout
        ), f"main.py not found in command:\n{result.stdout}"

    def test_model_inference(self, mlflow_client: MlflowClient, thesis_id: str) -> None:
        """Test that the registered model can run inference on test data.

        This test verifies:
        1. Model can be loaded from the registry
        2. Model can process test data and produce outputs
        3. Source run has correct thesis_id tag
        """
        try:
            versions = mlflow_client.search_model_versions(f"name='{thesis_id}'")
        except Exception:
            pytest.skip(f"Model '{thesis_id}' not registered")

        if not versions:
            pytest.skip(f"Model '{thesis_id}' not registered")

        # Load model from registry
        model_uri = f"models:/{thesis_id}/latest"
        try:
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

        # Get source run to load config
        latest = max(versions, key=lambda v: int(v.version))
        run_id = latest.run_id

        # Load config from run artifacts
        try:
            config_path = mlflow_client.download_artifacts(run_id, "config.yaml")
            from src.utils.config_manager import ConfigManager

            config = ConfigManager.from_yaml(config_path)
        except Exception as e:
            pytest.fail(f"Failed to load config: {e}")

        # Create datamodule and run inference
        from omegaconf import OmegaConf

        from src.data import get_datamodule
        from src.utils.training_utils import setup_metrics

        cfg = OmegaConf.create(
            {
                "dataset": config.dataset_config,
                "training": config.training_config,
            }
        )

        try:
            datamodule = get_datamodule(cfg)
            datamodule.setup(stage="test")
        except Exception as e:
            pytest.fail(f"Failed to setup datamodule: {e}")

        # Setup metrics and run test
        import pytorch_lightning as pl

        metric_names = config.model_config.get("metrics")
        metrics = setup_metrics(metric_names)
        model.setup_metrics(metrics)

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )

        try:
            results = trainer.test(model, datamodule=datamodule)
        except Exception as e:
            pytest.fail(f"Model inference failed: {e}")

        assert results is not None, "Test returned no results"
        assert len(results) > 0, "Test returned empty results"
        assert len(results[0]) > 0, "No metrics returned from test"

        # The reported metrics must reproduce: compare the freshly computed
        # test metrics against the values stored in the source run.
        # (Inference is deterministic up to GPU numerics -> tight tolerance.)
        stored = self._get_run_metrics(mlflow_client, run_id)
        recomputed = results[0]
        compared = 0
        for name, stored_value in stored.items():
            if name not in recomputed:
                continue
            compared += 1
            new_value = recomputed[name]
            rel_diff = (
                abs(new_value - stored_value) / abs(stored_value)
                if stored_value != 0
                else abs(new_value)
            )
            assert rel_diff <= 0.02, (
                f"Metric '{name}' does not reproduce:\n"
                f"  Stored at training time: {stored_value:.6f}\n"
                f"  Recomputed now:          {new_value:.6f}\n"
                f"  Difference: {rel_diff * 100:.2f}% (max 2%)\n"
                f"The registered model does not match the reported results."
            )
        assert compared > 0, (
            f"No overlapping test metrics to compare "
            f"(stored: {list(stored)}, recomputed: {list(recomputed)})"
        )

        # Verify thesis_id tag in source run
        run = mlflow_client.get_run(run_id)
        tags = run.data.tags

        assert "thesis_id" in tags, "Missing thesis_id tag in source run"
        assert (
            tags["thesis_id"] == thesis_id
        ), f"thesis_id mismatch: expected '{thesis_id}', got '{tags['thesis_id']}'"

    @pytest.mark.slow
    def test_reproduce_full(self, mlflow_client: MlflowClient, thesis_id: str) -> None:
        """Test full reproduction and compare metrics.

        WARNING: This test retrains the model from scratch and can take hours!
        Skip with: pytest -m "not slow"

        For quick validation, use test_model_inference instead.
        """
        run_id = self._get_source_run_id(mlflow_client, thesis_id)
        if not run_id:
            pytest.skip(f"Model '{thesis_id}' not registered")

        # Get original metrics
        original_metrics = self._get_run_metrics(mlflow_client, run_id)
        if not original_metrics:
            pytest.skip("No test metrics found in original run")

        tracking_uri = get_tracking_uri()

        # Run reproduction
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.utils.reproduce",
                run_id,
                "--tracking-uri",
                tracking_uri,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )

        assert result.returncode == 0, f"Reproduction failed:\n{result.stderr}"

        # Find the new run (most recent in same experiment)
        original_run = mlflow_client.get_run(run_id)
        experiment_id = original_run.info.experiment_id

        runs = mlflow_client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        assert len(runs) > 0, "No runs found after reproduction"
        new_run = runs[0]

        # Skip if it's the same run (shouldn't happen)
        if new_run.info.run_id == run_id:
            pytest.skip("No new run created")

        # Verify reproduction metadata in tags
        new_tags = new_run.data.tags
        assert (
            new_tags.get("is_reproduction") == "true"
        ), "Reproduced run missing 'is_reproduction' tag"
        assert new_tags.get("reproduced_from") == run_id, (
            f"Reproduced run has wrong 'reproduced_from' tag: "
            f"expected '{run_id}', got '{new_tags.get('reproduced_from')}'"
        )

        # Verify run name has [REPRODUCE] prefix
        run_name = new_tags.get("mlflow.runName", "")
        assert (
            "[REPRODUCE]" in run_name
        ), f"Run name should contain '[REPRODUCE]': {run_name}"

        # Compare metrics (within 20% tolerance for stochastic training)
        new_metrics = self._get_run_metrics(mlflow_client, new_run.info.run_id)

        for metric_name, original_value in original_metrics.items():
            if metric_name not in new_metrics:
                continue

            new_value = new_metrics[metric_name]
            tolerance = 0.20  # 20% tolerance

            # Calculate relative difference
            if original_value != 0:
                rel_diff = abs(new_value - original_value) / abs(original_value)
            else:
                rel_diff = abs(new_value)

            assert rel_diff <= tolerance, (
                f"Metric '{metric_name}' differs too much:\n"
                f"  Original: {original_value:.6f}\n"
                f"  Reproduced: {new_value:.6f}\n"
                f"  Difference: {rel_diff*100:.1f}% (max {tolerance*100:.0f}%)"
            )


class TestFigures:
    """Every thesis figure must be regenerable by scripts/make_figures.py."""

    def test_figures_script_runs(self, tmp_path) -> None:
        """The figure script runs end-to-end and produces output."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/make_figures.py",
                "--output",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"scripts/make_figures.py failed:\n{result.stderr}\n"
            f"Every thesis figure must be generated by this script."
        )

        produced = list(tmp_path.glob("*.pdf")) + list(tmp_path.glob("*.png"))
        assert produced, "make_figures.py ran but produced no figures"


class TestArchive:
    """The final model must be archived to the institute model store."""

    def test_archive_exists_and_is_complete(self, thesis_id: str) -> None:
        """models_root/<thesis_id>/ holds manifest, config, metrics, model."""
        from pathlib import Path

        from omegaconf import OmegaConf

        cfg = OmegaConf.load("configs/config.yaml")
        models_root = OmegaConf.select(cfg, "paths.models_root")
        archive_dir = Path(models_root) / thesis_id

        if not archive_dir.exists():
            pytest.fail(
                f"No submission archive at {archive_dir}.\n"
                f"Run the final step:\n"
                f"  uv run python -m src.utils.archive_submission"
            )

        for required in ("MANIFEST.json", "config.yaml", "metrics.json", "model"):
            assert (archive_dir / required).exists(), (
                f"Archive incomplete: '{required}' missing in {archive_dir}. "
                f"Rerun: uv run python -m src.utils.archive_submission --force"
            )

        import json

        manifest = json.loads((archive_dir / "MANIFEST.json").read_text())
        assert manifest["thesis_id"] == thesis_id
        assert not manifest["git_commit"].endswith("-dirty"), (
            "Archive was created from uncommitted changes. Commit your code, "
            "then rerun: uv run python -m src.utils.archive_submission --force"
        )
