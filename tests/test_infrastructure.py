"""
Infrastructure tests for the master thesis template.

These tests verify that the framework/infrastructure code works correctly:
- Dynamic model loading mechanism
- Dynamic data loading mechanism
- Config system

These are OPTIONAL tests for template maintainers.
Students should use tests/test_submission.py instead.

Run with: pytest tests/test_infrastructure.py
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.utils import import_class_from_path
from src.utils.model_registry import get_model_class


@pytest.fixture(scope="session")
def base_config():
    """Load base configuration with Hydra composition."""
    config_dir = str(Path("configs").absolute())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    return cfg


class TestModelRegistry:
    """Test model loading mechanism."""

    def test_import_class_from_path(self):
        """Test that class can be imported from dotted path."""
        from src.utils.model_registry import import_class_from_path

        # Test importing a known class
        class_path = "src.data.utils.import_class_from_path"
        imported = import_class_from_path(class_path)

        assert imported is not None
        assert callable(imported)

    def test_get_model_class(self, base_config):
        """Test that model class can be loaded from config."""
        model_name = base_config.model.name

        # Get model class
        model_class = get_model_class(model_name)

        assert model_class is not None, "Model class is None"
        assert hasattr(model_class, "__name__"), "Model class has no __name__"

    def test_model_config_has_class_path(self, base_config):
        """Test that model config has class_path field."""
        model_name = base_config.model.name
        model_cfg_path = Path(f"configs/model/{model_name}.yaml")

        assert model_cfg_path.exists(), f"Model config not found: {model_cfg_path}"

        model_cfg = OmegaConf.load(model_cfg_path)
        assert "class_path" in model_cfg, f"Missing 'class_path' in {model_name}.yaml"


class TestDataLoading:
    """Test data loading mechanism."""

    def test_import_datamodule_class(self):
        """Test that DataModule class can be imported."""
        # Test with a known datamodule class path
        class_path = "pytorch_lightning.LightningDataModule"
        imported = import_class_from_path(class_path)

        assert imported is not None
        assert hasattr(imported, "__name__")

    def test_data_config_has_datamodule_class(self, base_config):
        """Test that data config has datamodule_class in modality."""
        data_name = base_config.dataset.name
        modality = base_config.dataset.modality

        assert (
            "modalities" in base_config.dataset
        ), "Missing 'modalities' in data config"
        assert (
            modality in base_config.dataset.modalities
        ), f"Modality '{modality}' not in modalities"

        modality_cfg = base_config.dataset.modalities[modality]
        assert (
            "datamodule_class" in modality_cfg
        ), f"Missing 'datamodule_class' in {data_name}/{modality}"


class TestConfigSystem:
    """Test Hydra config system."""

    def test_base_config_structure(self, base_config):
        """Test that composed config has required fields."""
        assert "model" in base_config, "Missing 'model' in config"
        assert "dataset" in base_config, "Missing 'dataset' in config"
        assert "training" in base_config, "Missing 'training' in config"
        assert "paths" in base_config, "Missing 'paths' in config"

    def test_config_interpolation(self, base_config):
        """Test that config interpolations work."""
        # Test that submission_id is properly interpolated
        submission_id = base_config.submission.submission_id

        assert submission_id is not None
        assert "${" not in str(
            submission_id
        ), "Config interpolation failed, still contains ${...}"
