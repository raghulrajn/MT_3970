"""Shared fixtures for the test suite.

Two kinds of data fixtures:

- ``synthetic_ddacs_root``: a tiny fake DDACS root written into tmp_path.
  Unit tests run against this anywhere (CI, laptops) - no real data needed.
- ``ddacs_root``: the real dataset root from ``$DATA_DIR``. Tests using it
  must be marked ``@pytest.mark.slow`` and ``@requires_ddacs_data`` so they
  skip cleanly on machines without the data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DDACS_ROOT = Path(os.environ.get("DATA_DIR", "/mnt/data")) / "datasets" / "ddacs"

requires_ddacs_data = pytest.mark.skipif(
    not DDACS_ROOT.exists(),
    reason=f"real DDACS data not found at {DDACS_ROOT} (set DATA_DIR)",
)


@pytest.fixture
def ddacs_root() -> Path:
    """Root of the real DDACS dataset (guard tests with requires_ddacs_data)."""
    return DDACS_ROOT


@pytest.fixture
def synthetic_ddacs_root(tmp_path: Path) -> Path:
    """Minimal fake DDACS root: 4 sims, pointcloud/image exports, graphs.

    Layout matches the ddacs 3.x export formats:
    - pointcloud/ and image/: stacked .npy per field + sim_ids.npy + split.npy
    - graph/: one <sim_id>.npz per simulation
    - process_parameters.csv with the split column
    """
    rng = np.random.default_rng(0)
    sim_ids = np.array([101, 102, 103, 104])
    splits = np.array(["train", "train", "val", "test"])

    pd.DataFrame(
        {
            "index": sim_ids,
            "geometry": ["a", "a", "b", "b"],
            "curvature_radius": [1.0, 2.0, 3.0, 4.0],
            "bottom_radius": [5.0] * 4,
            "wall_angle": [30.0] * 4,
            "material_scaling_factor": [1.0] * 4,
            "sheet_metal_thickness": [1.5] * 4,
            "friction_coefficient": [0.1] * 4,
            "blankholder_force": [200.0] * 4,
            "split": splits,
        }
    ).to_csv(tmp_path / "process_parameters.csv", index=False)

    for subdir, fields in [
        ("pointcloud", {"blank": (4, 8, 3), "blank_thickness": (4, 8)}),
        ("image", {"blank_z": (4, 6, 6), "mask": (4, 6, 6)}),
    ]:
        d = tmp_path / subdir
        d.mkdir()
        np.save(d / "sim_ids.npy", sim_ids)
        np.save(d / "split.npy", splits)
        for name, shape in fields.items():
            np.save(d / f"{name}.npy", rng.random(shape, dtype=np.float32))

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    for sim_id, split in zip(sim_ids, splits):
        n_tool = int(rng.integers(3, 6))  # tool meshes vary in size
        np.savez(
            graph_dir / f"{sim_id}.npz",
            pos=rng.random((2, 5, 3), dtype=np.float32),  # (T, N, 3)
            edge_index=rng.integers(0, 5, (2, 7)),
            node_type=np.zeros(5, dtype=np.float32),
            binder_pos=rng.random((2, n_tool, 3), dtype=np.float32),
            binder_edge_index=rng.integers(0, n_tool, (2, 4)),
            norm_offset=rng.random(3, dtype=np.float32),
            blankholder_force=np.float64(200.0),
            split=split,
        )

    return tmp_path
