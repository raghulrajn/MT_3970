"""Model loading utilities.

Dynamically loads models from class_path specified in model configs.

STUDENT WORKFLOW:
1. Implement your model in src/models/your_model.py
2. Create configs/model/your_model.yaml with class_path field
3. Done!

Example config (configs/model/my_transformer.yaml):
    name: my_transformer
    class_path: src.models.my_transformer.MyTransformer
    architecture:
        ...
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Type

from omegaconf import DictConfig, OmegaConf

from src.models.base_model import BaseSurrogateModel

log = logging.getLogger(__name__)


def import_class_from_path(class_path: str) -> Type:
    """Dynamically import a class from a dotted path string.

    Args:
        class_path: Dotted path to class (e.g., "src.models.my_model.MyModel")

    Returns:
        The imported class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import class from path '{class_path}'. "
            f"Error: {e}\n"
            f"Make sure the path is correct: module.path.ClassName"
        )


def list_models() -> List[str]:
    """List all available models by scanning configs/model/ directory.

    Returns:
        List of model names (config file names without .yaml extension)
    """
    config_dir = Path("configs/model")
    if not config_dir.exists():
        log.warning(f"Model config directory not found: {config_dir}")
        return []

    models = []
    for yaml_file in config_dir.glob("*.yaml"):
        model_name = yaml_file.stem
        models.append(model_name)

    return sorted(models)


def get_model_class(model_name: str) -> Type[BaseSurrogateModel]:
    """Get model class by loading its config and importing from class_path.

    Args:
        model_name: Name of the model (corresponds to config file name)

    Returns:
        The model class

    Raises:
        FileNotFoundError: If model config not found
        ValueError: If class_path missing in config
        ImportError: If class cannot be imported
    """
    config_path = Path(f"configs/model/{model_name}.yaml")

    if not config_path.exists():
        available = list_models()
        raise FileNotFoundError(
            f"Model config not found: {config_path}\n" f"Available models: {available}"
        )

    cfg = OmegaConf.load(config_path)

    if "class_path" not in cfg:
        raise ValueError(
            f"Config {config_path} missing 'class_path' field.\n"
            f"Add: class_path: src.models.{model_name}.YourModelClass"
        )

    model_class = import_class_from_path(cfg.class_path)

    if not issubclass(model_class, BaseSurrogateModel):
        raise TypeError(
            f"Model class {cfg.class_path} must inherit from BaseSurrogateModel"
        )

    return model_class


def get_all_models() -> Dict[str, Type[BaseSurrogateModel]]:
    """Get all available models as a dictionary.

    Returns:
        Dictionary mapping model names to their classes
    """
    models = {}
    for model_name in list_models():
        try:
            models[model_name] = get_model_class(model_name)
        except Exception as e:
            log.warning(f"Could not load model '{model_name}': {e}")

    return models


def get_model(cfg: DictConfig) -> BaseSurrogateModel:
    """Create a model instance from config.

    Args:
        cfg: Hydra configuration with model.name and model.class_path

    Returns:
        Instantiated model (BaseSurrogateModel)

    Raises:
        ValueError: If model.name not specified
        ImportError: If model class cannot be imported

    Note:
        This is called automatically by the training script.
        Students don't need to call this function.
    """
    model_name = cfg.model.get("name")

    if model_name is None:
        raise ValueError(
            "No model specified in config. "
            "Usage: uv run main.py model=your_model dataset=ddacs"
        )

    # Get class_path from model config
    if "class_path" in cfg.model:
        class_path = cfg.model.class_path
    else:
        # Fall back to loading from config file
        model_cfg = OmegaConf.load(f"configs/model/{model_name}.yaml")
        if "class_path" not in model_cfg:
            raise ValueError(
                f"Model config 'configs/model/{model_name}.yaml' missing 'class_path' field.\n"
                f"Add: class_path: src.models.{model_name}.YourModelClass"
            )
        class_path = model_cfg.class_path

    # Import model class
    try:
        model_class = import_class_from_path(class_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import model class from '{class_path}'.\n"
            f"Make sure:\n"
            f"  1. File exists: {class_path.rsplit('.', 1)[0].replace('.', '/')}.py\n"
            f"  2. Class name is correct: {class_path.rsplit('.', 1)[1]}\n"
            f"  3. Class inherits from BaseSurrogateModel\n"
            f"Error: {e}"
        )

    # Convert OmegaConf to dict for model initialization
    model_config = {
        "name": model_name,
        "architecture": (
            dict(cfg.model.architecture) if "architecture" in cfg.model else {}
        ),
        "optimizer": dict(cfg.model.optimizer) if "optimizer" in cfg.model else {},
        "scheduler": dict(cfg.model.scheduler) if "scheduler" in cfg.model else {},
        "loss": dict(cfg.model.loss) if "loss" in cfg.model else {},
        "metrics": list(cfg.model.metrics) if "metrics" in cfg.model else [],
    }

    # Instantiate model
    return model_class(model_config)
