"""Mesh visualization utilities."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import SIZES, setup_plot_for_thesis

log = logging.getLogger(__name__)


def plot_meshes(
    sample: Dict[str, any],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
    edgecolor: str = "black",
    linewidth: float = 0.1,
    elev: float = 30,
    azim: float = 45,
    show: bool = True,
) -> plt.Figure:
    """Plot meshes for blank, die, punch, and binder.

    The blank is colored by thickness using the viridis colormap, while
    die, punch, and binder use solid colors.

    Args:
        sample: Dict containing PyG Data objects with keys 'blank', 'die',
            'punch', 'binder'. Each Data object must have attributes 'x' or 'pos'
            with shape (N, 3) containing vertex coordinates, and 'face' with
            shape (3, F) containing face indices. The blank should have 'face_attr'
            attribute containing thickness values per face.
        title: Optional title for the figure.
        save_path: Optional path to save the figure.
        figsize: Figure size as (width, height).
        edgecolor: Color of mesh edges.
        linewidth: Width of mesh edges.
        elev: Elevation viewing angle in degrees.
        azim: Azimuth viewing angle in degrees.
        show: Whether to display the plot.

    Returns:
        The matplotlib Figure object.

    Example:
        >>> import torch
        >>> from src.visualization.mesh import plot_meshes
        >>> blanks = torch.load("graph/blanks.pt", weights_only=False)
        >>> dies = torch.load("graph/dies.pt", weights_only=False)
        >>> punches = torch.load("graph/punches.pt", weights_only=False)
        >>> binders = torch.load("graph/binders.pt", weights_only=False)
        >>> sample = {
        ...     "blank": blanks[0],
        ...     "die": dies[0],
        ...     "punch": punches[0],
        ...     "binder": binders[0],
        ... }
        >>> plot_meshes(sample, title="Sample", save_path="output.png", show=False)
    """
    setup_plot_for_thesis()
    if figsize is None:
        figsize = SIZES["thesis"]["grid_2x2"]

    components = ["blank", "die", "punch", "binder"]
    colors = {
        "die": "#1f77b4",  # Blue
        "punch": "#2ca02c",  # Green
        "binder": "#EFBF04",  # Yellow
    }
    labels = {
        "blank": "Blank (thickness)",
        "die": "Die",
        "punch": "Punch",
        "binder": "Binder",
    }

    fig = plt.figure(figsize=figsize)

    for i, component in enumerate(components):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")

        data = sample.get(component)
        if data is None:
            ax.set_title(labels[component])
            log.warning(f"{component}: not available")
            continue

        if hasattr(data, "x"):
            vertices = data.x.cpu().numpy()
        elif hasattr(data, "pos"):
            vertices = data.pos.cpu().numpy()
        else:
            ax.set_title(labels[component])
            log.warning(f"{component}: no coordinates")
            continue

        if not hasattr(data, "face") or data.face is None:
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                c=colors.get(component, "gray"),
                s=1.0,
                alpha=1,
            )
            log.debug(f"{component}: {len(vertices)} vertices, no faces")
        else:
            faces = data.face.cpu().numpy().T  # (F, 3)

            if (
                component == "blank"
                and hasattr(data, "face_attr")
                and data.face_attr is not None
            ):
                face_thickness = data.face_attr.cpu().numpy().flatten()
                norm = Normalize(vmin=face_thickness.min(), vmax=face_thickness.max())
                # cmap = cm.viridis
                cmap = cm.jet_r
                face_colors = cmap(norm(face_thickness))

                mesh = Poly3DCollection(
                    vertices[faces],
                    alpha=1,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
                mesh.set_facecolor(face_colors)
                ax.add_collection3d(mesh)

                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
                cbar.set_label("Thickness")
            else:
                mesh = Poly3DCollection(
                    vertices[faces],
                    alpha=1.0,
                    facecolors=colors.get(component, "gray"),
                    edgecolors=edgecolor,
                    linewidth=linewidth,
                    shade=True,
                )
                ax.add_collection3d(mesh)

            log.debug(f"{component}: {len(vertices)} vertices, {len(faces)} faces")

        ax.set_title(labels[component])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        _set_equal_aspect(ax, vertices)
        ax.view_init(elev=elev, azim=azim)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.subplots_adjust(left=0.1, hspace=0.35, wspace=0)

    if save_path:
        plt.savefig(save_path, dpi=150)

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
    data_dir = Path(base_dir) / "datasets" / "ddacs" / "graph"

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

    plot_meshes(sample, title="Example Mesh", save_path="mesh_example.png", show=False)
    print("Saved: mesh_example.png")
