"""Automatic DataModule loading based on dataset configuration.

Similar to model loading, DataModules are loaded dynamically from class_path
specified in the data config. This module handles the translation from Hydra
config to explicit DataModule parameters.
"""

import importlib
import logging
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def import_class_from_path(class_path: str) -> type:
    """Dynamically import a class from a dotted path string.

    Args:
        class_path: Dotted path to class (e.g., "src.data.base_datamodules.GraphDataModule")

    Returns:
        The imported class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        datamodule_class = getattr(module, class_name)
        return datamodule_class
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import DataModule from path '{class_path}'. "
            f"Error: {e}\n"
            f"Make sure the path is correct: module.path.ClassName"
        )


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Automatically select and instantiate the correct DataModule.

    The function looks for datamodule_class in the selected modality config:
    cfg.dataset.modalities[cfg.dataset.modality].datamodule_class

    Args:
        cfg: Hydra configuration containing dataset.name and dataset.modality

    Returns:
        LightningDataModule instance

    Raises:
        ValueError: If datamodule_class not found in config
        ImportError: If class cannot be imported

    Example:
        # With cfg.dataset.name="ddacs" and cfg.dataset.modality="graph"
        # Config has: modalities.graphs.datamodule_class: src.data.base_datamodules.GraphDataModule
        datamodule = get_datamodule(cfg)
    """
    if "dataset" not in cfg:
        raise ValueError("Missing 'dataset' section in config")

    dataset_name: str | None = cfg.dataset.get("name")
    modality: str | None = cfg.dataset.get("modality")

    if not dataset_name:
        raise ValueError("Missing 'dataset.name' in config")
    if not modality:
        raise ValueError("Missing 'dataset.modality' in config")

    logger.info(f"Loading DataModule for dataset={dataset_name}, modality={modality}")

    # Get datamodule_class from modality config
    if "modalities" not in cfg.dataset or modality not in cfg.dataset.modalities:
        raise ValueError(
            f"Modality '{modality}' not found in dataset '{dataset_name}' config.\n"
            f"Available modalities: {list(cfg.dataset.get('modalities', {}).keys())}"
        )

    modality_cfg = cfg.dataset.modalities[modality]

    if "datamodule_class" not in modality_cfg:
        raise ValueError(
            f"Missing 'datamodule_class' in config for {dataset_name}/{modality}.\n"
            f"Add to configs/dataset/{dataset_name}.yaml:\n"
            f"  modalities:\n"
            f"    {modality}:\n"
            f"      datamodule_class: src.data.base_datamodules.GraphDataModule"
        )

    datamodule_class_path: str = modality_cfg.datamodule_class
    logger.info(f"Importing DataModule from: {datamodule_class_path}")

    # Import the DataModule class
    try:
        datamodule_class = import_class_from_path(datamodule_class_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import DataModule class from '{datamodule_class_path}'.\n"
            f"Make sure:\n"
            f"  1. File exists: {datamodule_class_path.rsplit('.', 1)[0].replace('.', '/')}.py\n"
            f"  2. Class name is correct: {datamodule_class_path.rsplit('.', 1)[1]}\n"
            f"  3. Class inherits from pl.LightningDataModule\n"
            f"Error: {e}"
        )

    # Extract parameters from config
    data_dir = modality_cfg.get("data_dir")
    if not data_dir:
        raise ValueError(f"Missing 'data_dir' in config for {dataset_name}/{modality}.")

    # Get root directory (parent of modality-specific data_dir)
    root = str(Path(data_dir).parent)

    batch_size = cfg.training.get("batch_size", 32)
    # dataloader.num_workers (training configs) wins over legacy num_workers
    dataloader_cfg = cfg.training.get("dataloader") or {}
    num_workers = dataloader_cfg.get("num_workers", cfg.training.get("num_workers", 4))

    # Optional modality-specific constructor kwargs (e.g. view: for streaming)
    extra_kwargs = modality_cfg.get("kwargs")
    extra_kwargs = OmegaConf.to_container(extra_kwargs) if extra_kwargs else {}

    # Instantiate the DataModule with explicit parameters
    try:
        datamodule = datamodule_class(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            **extra_kwargs,
        )
        logger.info(f"Successfully loaded {datamodule_class.__name__}")
        return datamodule
    except Exception as e:
        raise RuntimeError(
            f"Error instantiating {datamodule_class_path}.\n"
            f"Expected parameters: root, batch_size, num_workers"
            f"{', ' + ', '.join(extra_kwargs) if extra_kwargs else ''}.\n"
            f"Error: {e}"
        )
