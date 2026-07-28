"""Point cloud visualization utilities."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

from .config import SIZES, setup_plot_for_thesis

log = logging.getLogger(__name__)


def plot_pointclouds(
    sample: Dict[str, any],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
    point_size: float = 3.0,
    alpha: float = 0.6,
    elev: float = 30,
    azim: float = 45,
    show: bool = True,
) -> plt.Figure:
    """Plot point clouds for blank, die, punch, and binder.

    Args:
        sample: Dict containing PyG Data objects with keys 'blank', 'die',
            'punch', 'binder'. Each Data object must have attribute 'x'
            with shape (N, 3) containing point coordinates.
        title: Optional title for the figure.
        save_path: Optional path to save the figure.
        figsize: Figure size as (width, height).
        point_size: Size of points in scatter plot.
        alpha: Transparency of points.
        elev: Elevation viewing angle in degrees.
        azim: Azimuth viewing angle in degrees.
        show: Whether to display the plot.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> import torch
        >>> from src.visualization.pointcloud import plot_pointclouds
        >>> blanks = torch.load("pointcloud/blanks.pt", weights_only=False)
        >>> dies = torch.load("pointcloud/dies.pt", weights_only=False)
        >>> punches = torch.load("pointcloud/punches.pt", weights_only=False)
        >>> binders = torch.load("pointcloud/binders.pt", weights_only=False)
        >>> sample = {
        ...     "blank": blanks[0],
        ...     "die": dies[0],
        ...     "punch": punches[0],
        ...     "binder": binders[0],
        ... }
        >>> plot_pointclouds(sample, title="Sample", save_path="output.png", show=False)
    """
    setup_plot_for_thesis()
    if figsize is None:
        figsize = SIZES["thesis"]["grid_2x2"]

    components = ["blank", "die", "punch", "binder"]
    colors = {
        "blank": "#d62728",  # Red
        "die": "#1f77b4",  # Blue
        "punch": "#2ca02c",  # Green
        "binder": "#EFBF04",  # Yellow
    }
    labels = {
        "blank": "Blank",
        "die": "Die",
        "punch": "Punch",
        "binder": "Binder",
    }

    fig = plt.figure(figsize=figsize)

    for i, component in enumerate(components):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")

        data = sample.get(component)
        if data is None:
            ax.set_title(
                f"{labels[component]} (not available)",
            )
            continue

        if hasattr(data, "x"):
            points = data.x.cpu().numpy()
        elif hasattr(data, "pos"):
            points = data.pos.cpu().numpy()
        else:
            ax.set_title(labels[component])
            continue

        valid_mask = ~np.isnan(points).any(axis=1)
        valid_points = points[valid_mask]
        n_total = len(points)
        n_valid = len(valid_points)

        if n_valid < n_total:
            log.warning(f"{component}: {n_valid}/{n_total} valid points (NaN filtered)")
        else:
            log.debug(f"{component}: {n_total} points")

        ax.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            c=colors[component],
            s=point_size,
            alpha=alpha,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(labels[component])

        _set_equal_aspect(ax, points)
        ax.view_init(elev=elev, azim=azim)
        #
        ax.set_box_aspect(None, zoom=0.7)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.subplots_adjust(hspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)  # , bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _set_equal_aspect(ax: Axes3D, points: np.ndarray) -> None:
    """Set equal aspect ratio for 3D axes."""
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


if __name__ == "__main__":
    base_dir = os.environ.get("DATA_DIR", "/mnt/data")
    data_dir = Path(base_dir) / "datasets" / "ddacs" / "pointcloud"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Set DATA_DIR environment variable or ensure data exists at {data_dir}"
        )

    sample = {
        "blank": torch.load(data_dir / "blanks.pt", weights_only=False)[0],
        "die": torch.load(data_dir / "dies.pt", weights_only=False)[0],
        "punch": torch.load(data_dir / "punches.pt", weights_only=False)[0],
        "binder": torch.load(data_dir / "binders.pt", weights_only=False)[0],
    }

    plot_pointclouds(
        sample,
        title="Example Point Cloud",
        save_path="pointcloud_example.png",
        show=False,
    )
    print("Saved: pointcloud_example.png")
