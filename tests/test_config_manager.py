"""
Unit tests for ConfigManager.

Tests the configuration management system in isolation,
without requiring actual data directories or full Hydra initialization.

Run with: uv run pytest tests/test_config_manager.py -v
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.config_manager import ConfigManager


class TestConfigManagerInit:
    """Test ConfigManager initialization."""

    def test_init_with_minimal_config(self, tmp_path):
        """Test initialization with minimal required config."""
        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.base_data_dir == tmp_path
        assert cm.datasets_dir == tmp_path / "datasets"
        assert cm.models_dir == tmp_path / "models"

    def test_init_missing_base_data_dir_raises(self):
        """Test that missing base_data_dir raises ValueError."""
        config_dict = {"paths": {}}

        with pytest.raises(
            ValueError, match="Missing required config: paths.base_data_dir"
        ):
            ConfigManager(config_dict)

    def test_init_missing_paths_section_raises(self):
        """Test that missing paths section raises ValueError."""
        config_dict = {}

        with pytest.raises(
            ValueError, match="Missing required config: paths.base_data_dir"
        ):
            ConfigManager(config_dict)

    def test_thesis_id_computed_from_work_type_and_number(self, tmp_path):
        """Test thesis_id is computed as work_type_thesis_number."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "work_type": "MT",
            "thesis_number": "1234",
        }

        cm = ConfigManager(config_dict)

        assert cm.thesis_id == "MT_1234"

    def test_thesis_id_fallback_to_thesis_id_field(self, tmp_path):
        """Test thesis_id falls back to thesis_id field if work_type/number missing."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "thesis_id": "legacy_thesis_123",
        }

        cm = ConfigManager(config_dict)

        assert cm.thesis_id == "legacy_thesis_123"

    def test_student_metadata_stored(self, tmp_path):
        """Test student metadata is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "student_id": "12345",
            "student_name": "Test Student",
        }

        cm = ConfigManager(config_dict)

        assert cm.student_id == "12345"
        assert cm.student_name == "Test Student"

    def test_custom_datasets_and_models_root(self, tmp_path):
        """Test custom datasets_root and models_root paths."""
        datasets_path = tmp_path / "custom_datasets"
        models_path = tmp_path / "custom_models"

        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "datasets_root": str(datasets_path),
                "models_root": str(models_path),
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.datasets_dir == datasets_path
        assert cm.models_dir == models_path


class TestConfigManagerGet:
    """Test the get() method for accessing config values."""

    @pytest.fixture
    def config_manager(self, tmp_path):
        """Create a ConfigManager with nested config for testing."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "model": {
                "name": "test_mlp",
                "optimizer": {
                    "lr": 0.001,
                    "weight_decay": 1e-5,
                },
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
            },
        }
        return ConfigManager(config_dict)

    def test_get_simple_key(self, config_manager):
        """Test getting a simple top-level key."""
        training = config_manager.get("training")

        assert training["epochs"] == 100
        assert training["batch_size"] == 32

    def test_get_nested_key_dot_notation(self, config_manager):
        """Test getting nested value with dot notation."""
        lr = config_manager.get("model.optimizer.lr")

        assert lr == 0.001

    def test_get_deeply_nested_key(self, config_manager):
        """Test getting deeply nested value."""
        weight_decay = config_manager.get("model.optimizer.weight_decay")

        assert weight_decay == 1e-5

    def test_get_missing_key_returns_default(self, config_manager):
        """Test that missing key returns default value."""
        result = config_manager.get("nonexistent.key", default="fallback")

        assert result == "fallback"

    def test_get_missing_key_returns_none_by_default(self, config_manager):
        """Test that missing key returns None when no default specified."""
        result = config_manager.get("nonexistent.key")

        assert result is None

    def test_get_partial_path_missing(self, config_manager):
        """Test that partial path missing returns default."""
        result = config_manager.get("model.nonexistent.value", default=42)

        assert result == 42


class TestConfigManagerValidate:
    """Test the validate() method."""

    def test_validate_returns_dict(self, tmp_path):
        """Test validate returns dictionary with expected keys."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "work_type": "MT",
            "thesis_number": "2024_001",
        }

        cm = ConfigManager(config_dict)
        results = cm.validate()

        assert isinstance(results, dict)
        assert "base_dir_exists" in results
        assert "datasets_dir_exists" in results
        assert "models_dir_exists" in results
        assert "thesis_id_set" in results

    def test_validate_base_dir_exists(self, tmp_path):
        """Test validate detects existing base directory."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm = ConfigManager(config_dict)
        results = cm.validate()

        assert results["base_dir_exists"] is True

    def test_validate_base_dir_not_exists(self, tmp_path):
        """Test validate detects non-existing base directory."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path / "nonexistent")},
        }

        cm = ConfigManager(config_dict)
        results = cm.validate()

        assert results["base_dir_exists"] is False

    def test_validate_thesis_id_set(self, tmp_path):
        """Test validate detects thesis_id status."""
        config_with_thesis = {
            "paths": {"base_data_dir": str(tmp_path)},
            "work_type": "MT",
            "thesis_number": "2024_001",
        }
        config_without_thesis = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm_with = ConfigManager(config_with_thesis)
        cm_without = ConfigManager(config_without_thesis)

        assert cm_with.validate()["thesis_id_set"] is True
        assert cm_without.validate()["thesis_id_set"] is False


class TestConfigManagerToDict:
    """Test the to_dict() method."""

    def test_to_dict_basic_fields(self, tmp_path):
        """Test to_dict includes basic fields."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "student_id": "12345",
            "student_name": "Test Student",
            "work_type": "MT",
            "thesis_number": "2024_001",
        }

        cm = ConfigManager(config_dict)
        result = cm.to_dict()

        assert result["base_data_dir"] == str(tmp_path)
        assert result["student_id"] == "12345"
        assert result["student_name"] == "Test Student"
        assert result["thesis_id"] == "MT_2024_001"


class TestConfigManagerRepr:
    """Test string representation."""

    def test_repr_includes_student_info(self, tmp_path):
        """Test __repr__ includes student information."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "student_name": "Test Student",
            "student_id": "12345",
            "work_type": "MT",
            "thesis_number": "2024_001",
        }

        cm = ConfigManager(config_dict)
        repr_str = repr(cm)

        assert "ConfigManager" in repr_str
        assert "Test Student" in repr_str
        assert "12345" in repr_str
        assert "MT_2024_001" in repr_str

    def test_repr_handles_no_student_info(self, tmp_path):
        """Test __repr__ handles missing student information."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm = ConfigManager(config_dict)
        repr_str = repr(cm)

        assert "ConfigManager" in repr_str
        assert "No student info" in repr_str


class TestConfigManagerSubmission:
    """Test submission-related functionality."""

    def test_submission_id_from_config(self, tmp_path):
        """Test submission_id is read from config when present."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "submission": {
                "submission_id": "MT_2024_001_mlp_ddacs",
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.submission_id == "MT_2024_001_mlp_ddacs"

    def test_submission_id_computed_fallback(self, tmp_path):
        """Test submission_id is computed when not in config."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "work_type": "MT",
            "thesis_number": "2024_001",
            "model": {"name": "my_model"},
            "dataset": {"name": "ddacs"},
        }

        cm = ConfigManager(config_dict)

        assert cm.submission_id == "MT_2024_001_my_model_ddacs"


class TestConfigManagerDatasetPath:
    """Test get_dataset_path() and dataset_path property."""

    def test_get_dataset_path_returns_correct_path(self, tmp_path):
        """Test get_dataset_path returns the configured data_dir."""
        # Create the dataset directory
        dataset_dir = tmp_path / "datasets" / "ddacs" / "h5"
        dataset_dir.mkdir(parents=True)

        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "h5",
                "modalities": {
                    "h5": {
                        "datamodule_class": "some.module.H5DataModule",
                        "data_dir": str(dataset_dir),
                    },
                },
            },
        }

        cm = ConfigManager(config_dict)
        result = cm.get_dataset_path()

        assert result == dataset_dir

    def test_get_dataset_path_missing_data_section_raises(self, tmp_path):
        """Test get_dataset_path raises when data section missing."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(ValueError, match="Missing 'dataset' section"):
            cm.get_dataset_path()

    def test_get_dataset_path_missing_name_raises(self, tmp_path):
        """Test get_dataset_path raises when dataset.name missing."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "modality": "h5",
            },
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(ValueError, match="Missing 'dataset.name'"):
            cm.get_dataset_path()

    def test_get_dataset_path_missing_modality_raises(self, tmp_path):
        """Test get_dataset_path raises when dataset.modality missing."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
            },
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(ValueError, match="Missing 'dataset.modality'"):
            cm.get_dataset_path()

    def test_get_dataset_path_invalid_modality_raises(self, tmp_path):
        """Test get_dataset_path raises when modality not in modalities."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "nonexistent",
                "modalities": {
                    "h5": {"data_dir": str(tmp_path)},
                },
            },
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            cm.get_dataset_path()

    def test_get_dataset_path_missing_data_dir_raises(self, tmp_path):
        """Test get_dataset_path raises when data_dir missing in modality."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "h5",
                "modalities": {
                    "h5": {
                        "datamodule_class": "some.module.H5DataModule",
                        # data_dir is missing
                    },
                },
            },
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(ValueError, match="Missing 'data_dir'"):
            cm.get_dataset_path()

    def test_get_dataset_path_nonexistent_path_raises(self, tmp_path):
        """Test get_dataset_path raises FileNotFoundError for nonexistent path."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "h5",
                "modalities": {
                    "h5": {
                        "data_dir": str(tmp_path / "nonexistent" / "path"),
                    },
                },
            },
        }

        cm = ConfigManager(config_dict)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            cm.get_dataset_path()

    def test_get_dataset_path_multiple_modalities(self, tmp_path):
        """Test get_dataset_path works with multiple modalities configured."""
        h5_dir = tmp_path / "datasets" / "ddacs" / "h5"
        graph_dir = tmp_path / "datasets" / "ddacs" / "graphs"
        h5_dir.mkdir(parents=True)
        graph_dir.mkdir(parents=True)

        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "graphs",  # Select graphs modality
                "modalities": {
                    "h5": {"data_dir": str(h5_dir)},
                    "graphs": {"data_dir": str(graph_dir)},
                },
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.get_dataset_path() == graph_dir


class TestConfigManagerListStudentModels:
    """Test list_student_models() method."""

    def test_list_student_models_empty(self, tmp_path):
        """Test list_student_models returns empty list when no models."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "models_root": str(models_dir),
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.list_student_models() == []

    def test_list_student_models_returns_directories(self, tmp_path):
        """Test list_student_models returns directory names."""
        models_dir = tmp_path / "models"
        (models_dir / "MT_2024_001").mkdir(parents=True)
        (models_dir / "MT_2024_002").mkdir(parents=True)
        (models_dir / "BT_2024_003").mkdir(parents=True)

        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "models_root": str(models_dir),
            },
        }

        cm = ConfigManager(config_dict)
        result = cm.list_student_models()

        assert len(result) == 3
        assert "MT_2024_001" in result
        assert "MT_2024_002" in result
        assert "BT_2024_003" in result

    def test_list_student_models_ignores_files(self, tmp_path):
        """Test list_student_models ignores files, only returns directories."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "MT_2024_001").mkdir()
        (models_dir / "readme.txt").touch()  # File, not directory

        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "models_root": str(models_dir),
            },
        }

        cm = ConfigManager(config_dict)
        result = cm.list_student_models()

        assert result == ["MT_2024_001"]

    def test_list_student_models_nonexistent_dir(self, tmp_path):
        """Test list_student_models returns empty list when models_dir doesn't exist."""
        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "models_root": str(tmp_path / "nonexistent_models"),
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.list_student_models() == []


class TestConfigManagerMLflow:
    """Test MLflow-related methods."""

    def test_get_mlflow_tracking_uri_format(self, tmp_path):
        """Test get_mlflow_tracking_uri returns sqlite:// URI."""
        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "experiments_dir": str(tmp_path / "experiments"),
            },
        }

        cm = ConfigManager(config_dict)
        uri = cm.get_mlflow_tracking_uri()

        assert uri.startswith("sqlite:///")
        assert "mlflow.db" in uri

    def test_get_mlflow_tracking_uri_creates_directory(self, tmp_path):
        """Test get_mlflow_tracking_uri creates mlflow directory."""
        config_dict = {
            "paths": {
                "base_data_dir": str(tmp_path),
                "experiments_dir": str(tmp_path / "experiments"),
            },
        }

        cm = ConfigManager(config_dict)
        cm.get_mlflow_tracking_uri()

        assert cm.mlflow_dir.exists()

    def test_mlflow_dir_default(self, tmp_path):
        """Test mlflow_dir defaults to ./mlruns."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm = ConfigManager(config_dict)

        assert cm.mlflow_dir == Path("./mlruns")


class TestConfigManagerHydraIntegration:
    """Test Hydra integration with from_hydra() classmethod."""

    def test_from_hydra_with_dictconfig(self, tmp_path):
        """Test from_hydra creates ConfigManager from DictConfig."""

        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "student_name": "Test Student",
            "work_type": "MT",
            "thesis_number": "2024_001",
        }
        hydra_cfg = OmegaConf.create(config_dict)

        cm = ConfigManager.from_hydra(hydra_cfg)

        assert cm.base_data_dir == tmp_path
        assert cm.student_name == "Test Student"
        assert cm.thesis_id == "MT_2024_001"

    def test_from_hydra_resolves_interpolations(self, tmp_path):
        """Test from_hydra resolves OmegaConf interpolations."""
        from omegaconf import OmegaConf

        config_dict = {
            "base_path": str(tmp_path),
            "paths": {
                "base_data_dir": "${base_path}",
                "datasets_root": "${base_path}/datasets",
            },
        }
        hydra_cfg = OmegaConf.create(config_dict)

        cm = ConfigManager.from_hydra(hydra_cfg)

        assert cm.base_data_dir == tmp_path
        assert cm.datasets_dir == tmp_path / "datasets"

    def test_from_hydra_with_nested_config(self, tmp_path):
        """Test from_hydra handles nested configuration."""
        from omegaconf import OmegaConf

        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "model": {
                "name": "test_mlp",
                "architecture": {
                    "hidden_dims": [64, 32],
                    "activation": "relu",
                },
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
            },
        }
        hydra_cfg = OmegaConf.create(config_dict)

        cm = ConfigManager.from_hydra(hydra_cfg)

        assert cm.get("model.name") == "test_mlp"
        assert cm.get("model.architecture.hidden_dims") == [64, 32]
        assert cm.get("training.epochs") == 100


class TestConfigManagerSubConfigs:
    """Test access to raw sub-configs."""

    def test_model_config_accessible(self, tmp_path):
        """Test model_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "model": {
                "name": "test_mlp",
                "hidden_dims": [128, 64],
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.model_config["name"] == "test_mlp"
        assert cm.model_config["hidden_dims"] == [128, 64]

    def test_dataset_config_accessible(self, tmp_path):
        """Test dataset_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "dataset": {
                "name": "ddacs",
                "modality": "h5",
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.dataset_config["name"] == "ddacs"
        assert cm.dataset_config["modality"] == "h5"

    def test_training_config_accessible(self, tmp_path):
        """Test training_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "training": {
                "epochs": 100,
                "batch_size": 32,
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.training_config["epochs"] == 100
        assert cm.training_config["batch_size"] == 32

    def test_logging_config_accessible(self, tmp_path):
        """Test logging_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "logging": {
                "logger": "mlflow",
                "log_every_n_steps": 10,
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.logging_config["logger"] == "mlflow"

    def test_checkpoint_config_accessible(self, tmp_path):
        """Test checkpoint_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "checkpoint": {
                "save_top_k": 3,
                "monitor": "val/loss",
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.checkpoint_config["save_top_k"] == 3
        assert cm.checkpoint_config["monitor"] == "val/loss"

    def test_early_stopping_config_accessible(self, tmp_path):
        """Test early_stopping_config attribute is accessible."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
            "early_stopping": {
                "enabled": True,
                "patience": 15,
            },
        }

        cm = ConfigManager(config_dict)

        assert cm.early_stopping_config["enabled"] is True
        assert cm.early_stopping_config["patience"] == 15

    def test_missing_subconfigs_return_empty_dict(self, tmp_path):
        """Test missing sub-configs return empty dicts."""
        config_dict = {
            "paths": {"base_data_dir": str(tmp_path)},
        }

        cm = ConfigManager(config_dict)

        assert cm.model_config == {}
        assert cm.dataset_config == {}
        assert cm.training_config == {}
        assert cm.logging_config == {}
        assert cm.checkpoint_config == {}
        assert cm.early_stopping_config == {}
