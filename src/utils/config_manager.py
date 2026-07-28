"""Unified configuration management for paths, datasets, and experiment settings.

This module provides a ConfigManager that works exclusively with Hydra/OmegaConf configs.

Usage:
    # Use with Hydra config
    @hydra.main(config_path="configs", config_name="config")
    def main(cfg):
        config = ConfigManager.from_hydra(cfg)

    # Access configuration
    dataset_path = config.get_dataset_path()
    tracking_uri = config.get_mlflow_tracking_uri()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import yaml
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """Configuration manager for paths, datasets, and settings.

    Works exclusively with Hydra/OmegaConf configs.

    Attributes:
        base_data_dir: Base directory for centralized data storage.
        datasets_dir: Directory containing datasets.
        models_dir: Directory containing student models.
        mlflow_dir: Directory for MLflow tracking (single source of truth).
        thesis_id: Thesis identifier (optional, for logging).
        student_id: Student ID (optional, for logging).
        student_name: Student name (optional, for logging).
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize ConfigManager from Hydra config dict.

        Args:
            config_dict: Configuration dictionary from OmegaConf.to_container().
        """
        self._config = config_dict

        if "paths" in self._config and "base_data_dir" in self._config["paths"]:
            self.base_data_dir = Path(self._config["paths"]["base_data_dir"])
        else:
            raise ValueError(
                "Missing required config: paths.base_data_dir. "
                "Please define it in your config.yaml (can use ${oc.env:DATA_DIR,/default/path})"
            )

        self.student_id = self._config.get("student_id")
        self.student_name = self._config.get("student_name")

        work_type = self._config.get("work_type")
        thesis_number = self._config.get("thesis_number")

        # Validate student information is configured
        self._validate_student_info(work_type, thesis_number)

        if work_type and thesis_number:
            self.thesis_id = f"{work_type}_{thesis_number}"
        else:
            self.thesis_id = self._config.get("thesis_id")

        if (
            "submission" in self._config
            and "submission_id" in self._config["submission"]
        ):
            self.submission_id = self._config["submission"]["submission_id"]
        else:
            model_name = self._config.get("model", {}).get("name", "unknown")
            data_name = self._config.get("dataset", {}).get("name", "unknown")
            self.submission_id = f"{self.thesis_id}_{model_name}_{data_name}"

        if "paths" in self._config:
            paths_cfg = self._config["paths"]
            self.datasets_dir = Path(
                paths_cfg.get("datasets_root", self.base_data_dir / "datasets")
            )
            self.models_dir = Path(
                paths_cfg.get("models_root", self.base_data_dir / "models")
            )
        else:
            self.datasets_dir = self.base_data_dir / "datasets"
            self.models_dir = self.base_data_dir / "models"

        self.mlflow_dir = Path("./mlruns")

        self.model_config = self._config.get("model", {})
        self.dataset_config = self._config.get("dataset", {})
        self.training_config = self._config.get("training", {})
        self.logging_config = self._config.get("logging", {})
        self.checkpoint_config = self._config.get("checkpoint", {})
        self.early_stopping_config = self._config.get("early_stopping", {})

    def _validate_student_info(
        self, work_type: Optional[str], thesis_number: Optional[str]
    ) -> None:
        """Validate that student information has been configured.

        Raises:
            ValueError: If any student info field is still set to "CHANGE_ME".
        """
        fields_to_check = {
            "student_name": self.student_name,
            "student_id": self.student_id,
            "work_type": work_type,
            "thesis_number": thesis_number,
        }

        unconfigured = [
            name for name, value in fields_to_check.items() if value == "CHANGE_ME"
        ]

        if unconfigured:
            raise ValueError(
                f"Student information not configured in configs/config.yaml!\n\n"
                f"Please update the following fields: {', '.join(unconfigured)}\n\n"
                f"Open configs/config.yaml and replace 'CHANGE_ME' with your actual values:\n"
                f'  student_name: "Your Full Name"\n'
                f'  student_id: "st123456"\n'
                f'  work_type: "MT"  # BT, MT, or RT\n'
                f'  thesis_number: "2024_042"\n'
            )

    @classmethod
    def from_hydra(cls, hydra_cfg: DictConfig) -> "ConfigManager":
        """Create ConfigManager from Hydra/OmegaConf config.

        Args:
            hydra_cfg: Hydra DictConfig object.

        Returns:
            Initialized ConfigManager instance.

        Example:
            @hydra.main(config_path="configs", config_name="config")
            def main(cfg):
                config = ConfigManager.from_hydra(cfg)
        """
        config_dict = OmegaConf.to_container(hydra_cfg, resolve=True)

        # Get runtime info from HydraConfig singleton (more reliable than config extraction)
        try:
            hydra_cfg_singleton = HydraConfig.get()
            if hydra_cfg_singleton is not None:
                hydra_dict = {
                    "run": {
                        "dir": hydra_cfg_singleton.runtime.output_dir,
                    },
                    "runtime": {
                        "output_dir": hydra_cfg_singleton.runtime.output_dir,
                        "cwd": hydra_cfg_singleton.runtime.cwd,
                    },
                }
                config_dict["hydra"] = hydra_dict  # type: ignore
        except Exception:
            # HydraConfig not initialized - not running with Hydra
            pass

        return cls(config_dict=config_dict)  # type: ignore

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ConfigManager":
        """Create ConfigManager from a YAML file.

        Useful for loading config from MLflow artifacts.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            Initialized ConfigManager instance.

        Example:
            config = ConfigManager.from_yaml("path/to/config.yaml")
        """
        cfg = OmegaConf.load(yaml_path)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls(config_dict=config_dict)  # type: ignore

    @property
    def is_reproduction(self) -> bool:
        """Check if this is a reproduction run.

        Returns:
            True if this run was started via reproduce.py.
        """
        return self._config.get("reproduction", {}).get("is_reproduction", False)

    @property
    def reproduced_from(self) -> Optional[str]:
        """Get the original run_id this is reproducing.

        Returns:
            Original run_id or None if not a reproduction.
        """
        return self._config.get("reproduction", {}).get("reproduced_from")

    @property
    def run_name(self) -> Optional[str]:
        """Get MLflow run name (basename of experiment dir).

        Returns:
            Run name string (e.g., "dummy_ddacs") or None if model/data not set.
            Prefixed with [REPRODUCE] if this is a reproduction run.

        Example:
            config = ConfigManager.from_hydra(cfg)
            run_name = config.run_name  # "dummy_ddacs" or "[REPRODUCE] dummy_ddacs"
        """
        model_name = self.model_config.get("name")
        data_name = self.dataset_config.get("name")
        if model_name and data_name:
            base_name = f"{model_name}_{data_name}"
            if self.is_reproduction:
                return f"[REPRODUCE] {base_name}"
            return base_name
        return None

    @property
    def mlflow_tags(self) -> Dict[str, str]:
        """Get tags for MLflow run.

        Returns:
            Dictionary of tags for MLflow tracking including model_name,
            dataset_name, student info, and reproduction metadata if available.

        Example:
            config = ConfigManager.from_hydra(cfg)
            tags = config.mlflow_tags
            # {'model_name': 'dummy', 'dataset_name': 'ddacs', ...}
        """
        tags = {}
        if self.model_config.get("name"):
            tags["model_name"] = self.model_config.get("name")
        if self.dataset_config.get("name"):
            tags["dataset_name"] = self.dataset_config.get("name")
        if self.student_name:
            tags["student_name"] = self.student_name
        if self.thesis_id:
            tags["thesis_id"] = self.thesis_id
        if self.is_reproduction:
            tags["is_reproduction"] = "true"
            if self.reproduced_from:
                tags["reproduced_from"] = self.reproduced_from
        return tags

    def get_dataset_path(self) -> Path:
        """Get path to the configured dataset modality.

        Reads dataset name and modality from the stored Hydra config
        (cfg.dataset.name and cfg.dataset.modality) and returns the corresponding
        data directory path.

        Returns:
            Path object pointing to the dataset modality directory.

        Raises:
            ValueError: If dataset.name, dataset.modality, or data_dir is not configured.
            FileNotFoundError: If dataset directory does not exist.

        Example:
            # With config: dataset.name=ddacs, +dataset.modality=graph
            config = ConfigManager.from_hydra(cfg)
            dataset_path = config.get_dataset_path()
            # Returns: /mnt/data/datasets/ddacs/graphs
        """
        if "dataset" not in self._config:
            raise ValueError("Missing 'dataset' section in config.")

        dataset_cfg = self._config["dataset"]
        dataset_name = dataset_cfg.get("name")
        modality = dataset_cfg.get("modality")

        if not dataset_name:
            raise ValueError("Missing 'dataset.name' in config.")
        if not modality:
            raise ValueError(
                "Missing 'dataset.modality' in config. "
                "Specify modality in dataset config or override with: uv run main.py +dataset.modality=h5"
            )

        if "modalities" not in dataset_cfg or modality not in dataset_cfg["modalities"]:
            available_modalities = list(dataset_cfg.get("modalities", {}).keys())
            raise ValueError(
                f"Modality '{modality}' not found in dataset '{dataset_name}'. "
                f"Available modalities: {available_modalities}"
            )

        modality_config = dataset_cfg["modalities"][modality]
        data_dir = modality_config.get("data_dir")

        if not data_dir:
            raise ValueError(
                f"Missing 'data_dir' for modality '{modality}' in dataset '{dataset_name}'."
            )

        dataset_path = Path(data_dir)

        if not dataset_path.exists():
            available = self.list_available_datasets()
            raise FileNotFoundError(
                f"Dataset path '{dataset_path}' does not exist. "
                f"Dataset: {dataset_name}, Modality: {modality}. "
                f"Available datasets: {available}"
            )

        return dataset_path

    def get_mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI using SQLite backend.

        Returns:
            MLflow tracking URI string with SQLite database.

        Example:
            mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())
        """
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.mlflow_dir.absolute()}/mlflow.db"

    def list_available_datasets(self) -> list:
        """List all available datasets with modalities.

        Checks all configured datasets from configs/dataset/*.yaml and verifies
        which modalities actually exist on disk.

        Returns:
            List of dataset/modality pairs (e.g., ['ddacs/h5', 'ddacs/graphs']).
        """
        available = []

        if not GlobalHydra.instance().is_initialized():
            return available

        config_search_path = GlobalHydra.instance().config_loader().get_search_path()
        config_dir = None

        for search_path in config_search_path.config_search_path:  # type: ignore
            if search_path.provider == "main":
                config_dir = Path(search_path.path) / "dataset"
                break

        if not config_dir or not config_dir.exists():
            return available

        for config_file in config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                dataset_config = yaml.safe_load(f)

            if not dataset_config or "modalities" not in dataset_config:
                continue

            dataset_name = dataset_config.get("name", config_file.stem)
            modalities = dataset_config["modalities"]

            for modality_name, modality_config in modalities.items():
                if "data_dir" in modality_config:
                    data_dir_template = modality_config["data_dir"]
                    if (
                        "paths" in self._config
                        and "datasets_root" in self._config["paths"]
                    ):
                        datasets_root = self._config["paths"]["datasets_root"]
                        data_dir_str = data_dir_template.replace(
                            "${paths.datasets_root}", datasets_root
                        )
                    else:
                        data_dir_str = data_dir_template

                    data_dir = Path(data_dir_str)
                    if data_dir.exists():
                        available.append(f"{dataset_name}/{modality_name}")

        return available

    def list_student_models(self) -> list:
        """List all student model directories.

        Returns:
            List of thesis IDs/student identifiers.
        """
        if not self.models_dir.exists():
            return []

        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports nested keys using dot notation.

        Args:
            key: Configuration key (e.g., 'model.optimizer.lr').
            default: Default value if key not found.

        Returns:
            Configuration value.

        Example:
            lr = config.get("model.optimizer.lr", 0.001)
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate(self) -> Dict[str, bool]:
        """Validate configuration and paths.

        Returns:
            Dictionary with validation results.

        Example:
            results = config.validate()
            if not results["base_dir_exists"]:
                print("Error: Base directory not found!")
        """
        results = {
            "base_dir_exists": self.base_data_dir.exists(),
            "base_dir_writable": (
                os.access(self.base_data_dir, os.W_OK)
                if self.base_data_dir.exists()
                else False
            ),
            "datasets_dir_exists": self.datasets_dir.exists(),
            "datasets_dir_readable": (
                os.access(self.datasets_dir, os.R_OK)
                if self.datasets_dir.exists()
                else False
            ),
            "models_dir_exists": self.models_dir.exists(),
            "models_dir_writable": (
                os.access(self.models_dir, os.W_OK)
                if self.models_dir.exists()
                else False
            ),
            "thesis_id_set": self.thesis_id is not None,
            "available_datasets": self.list_available_datasets(),
            "student_models": self.list_student_models(),
        }

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "base_data_dir": str(self.base_data_dir),
            "datasets_dir": str(self.datasets_dir),
            "models_dir": str(self.models_dir),
            "mlflow_dir": str(self.mlflow_dir),
            "thesis_id": self.thesis_id,
            "student_id": self.student_id,
            "student_name": self.student_name,
            "config": self._config,
        }

    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        student_info = []
        if self.student_name:
            student_info.append(f"student_name={self.student_name}")
        if self.student_id:
            student_info.append(f"student_id={self.student_id}")
        if self.thesis_id:
            student_info.append(f"thesis_id={self.thesis_id}")

        student_str = ", ".join(student_info) if student_info else "No student info"

        return (
            f"ConfigManager(\n"
            f"  base_data_dir={self.base_data_dir},\n"
            f"  student_info=({student_str}),\n"
            f"  datasets={len(self.list_available_datasets())},\n"
            f"  models={len(self.list_student_models())}\n"
            f")"
        )


# Convenience function for Hydra configs
def get_config(hydra_cfg: DictConfig) -> ConfigManager:
    """Get ConfigManager instance from Hydra config.

    Args:
        hydra_cfg: Hydra DictConfig object.

    Returns:
        Initialized ConfigManager instance.

    Example:
        @hydra.main(config_path="configs", config_name="config")
        def main(cfg):
            config = get_config(cfg)
    """
    return ConfigManager.from_hydra(hydra_cfg)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Print configuration and validation results."""
    print("ConfigManager Validation")
    print("=" * 70)

    config = ConfigManager.from_hydra(cfg)
    print(config)
    print()

    print("Student Information:")
    print(f"  Student Name: {config.student_name or 'Not set'}")
    print(f"  Student ID: {config.student_id or 'Not set'}")
    print(f"  Thesis ID: {config.thesis_id or 'Not set'}")
    print()

    print("Paths:")
    print(f"  Base data dir: {config.base_data_dir}")
    print(f"  Datasets dir: {config.datasets_dir}")
    print(f"  Models dir: {config.models_dir}")
    print(f"  MLflow URI: {config.get_mlflow_tracking_uri()}")
    print()

    print("Validation Results:")
    print("-" * 70)
    results = config.validate()

    for key, value in results.items():
        if isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            status = "OK" if value else "FAIL"
            print(f"  {key}: [{status}]")

    print()

    if results["base_dir_exists"]:
        print("Available datasets:")
        # Group datasets by base dataset name
        dataset_tree = {}
        for dataset in results["available_datasets"]:  # type: ignore
            if "/" in dataset:
                base, modality = dataset.split("/", 1)
                if base not in dataset_tree:
                    dataset_tree[base] = []
                dataset_tree[base].append(modality)
            else:
                dataset_tree[dataset] = []

        # Print in tree structure
        for base_dataset in sorted(dataset_tree.keys()):
            print(f"  {base_dataset}/")
            modalities = dataset_tree[base_dataset]
            for i, modality in enumerate(sorted(modalities)):
                is_last = i == len(modalities) - 1
                prefix = "    └── " if is_last else "    ├── "
                print(f"{prefix}{modality}")
        print()

        print("Student model directories:")
        for student in results["student_models"]:  # type: ignore
            print(f"  - {student}")
    else:
        print(f"ERROR: Base directory {config.base_data_dir} does not exist!")
        print("Set environment variable: export THESIS_DATA_DIR=/path/to/data")
        print("Or run setup script: sudo bash setup_server.sh")


if __name__ == "__main__":
    main()
