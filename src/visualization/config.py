"""Plotting configuration for consistent figure styles."""

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

SIZES = {
    "thesis": {
        "full": (6.3, 2.1),
        "half": (3.0, 2.1),
        "grid_2x2": (6.3, 5.0),
    },
    "paper": {
        "single_col": (3.5, 2.5),
        "double_col": (7.16, 2.5),
        "grid_2x2": (7.16, 5.0),
    },
}


def setup_plot_for_thesis():
    """Setup matplotlib for thesis figures (DIN A4 single column)."""
    plt.style.use(["science", "no-latex"])
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def setup_plot_for_paper():
    """Setup matplotlib for IEEE paper figures (two-column)."""
    plt.style.use(["science", "ieee", "no-latex"])
