"""File utility functions for the master thesis template.

Simple utility functions for file operations that don't require Hydra initialization.
"""

from pathlib import Path


def list_dataset_configs(config_dir: Path = Path("configs/dataset")) -> list[str]:
    """Get list of all available dataset names from config files.

    Args:
        config_dir: Path to the data config directory.

    Returns:
        Sorted list of dataset names (config file stems).

    Example:
        >>> datasets = list_dataset_configs()
        >>> print(datasets)
        ['ddacs', 'hsh']
    """
    if not config_dir.exists():
        return []

    datasets = []
    for yaml_file in config_dir.glob("*.yaml"):
        datasets.append(yaml_file.stem)

    return sorted(datasets)
