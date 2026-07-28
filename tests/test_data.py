"""Tests for the data loading system (ddacs 3.x based).

Unit tests run against a synthetic dataset (see conftest.py) and need no
real data. Integration tests (marked ``slow`` + skipped without data)
pull one batch from every real modality.

Run with:
    uv run pytest tests/test_data.py -v              # unit tests
    uv run pytest tests/test_data.py -v -m slow      # + real-data tests
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.data.ddacs import (
    PARAM_COLUMNS,
    GraphDDACS,
    GraphDDACSDataset,
    ImageDDACS,
    PointCloudDDACS,
    PointCloudDDACSDataset,
    StreamingDDACS,
    graph_collate_fn,
)
from src.data.utils import get_datamodule, import_class_from_path
from tests.conftest import requires_ddacs_data


class TestImportClassFromPath:
    """Test import_class_from_path utility function."""

    def test_import_builtin_class(self):
        """Test importing a standard library class."""
        from pathlib import Path

        assert import_class_from_path("pathlib.Path") is Path

    def test_import_local_class(self):
        """Test importing a local project class."""
        assert import_class_from_path("src.data.ddacs.StreamingDDACS") is StreamingDDACS

    def test_import_invalid_module_raises(self):
        """Test that invalid module path raises ImportError."""
        with pytest.raises(ImportError, match="Could not import"):
            import_class_from_path("nonexistent.module.Class")

    def test_import_invalid_class_raises(self):
        """Test that invalid class name raises ImportError."""
        with pytest.raises(ImportError, match="Could not import"):
            import_class_from_path("pathlib.NonexistentClass")

    def test_import_malformed_path_raises(self):
        """Test that malformed path raises ImportError."""
        with pytest.raises(ImportError, match="Could not import"):
            import_class_from_path("nomodule")


class TestGetDatamodule:
    """Test get_datamodule utility function."""

    def test_missing_dataset_section_raises(self):
        """Test that missing dataset section raises ValueError."""
        with pytest.raises(ValueError, match="Missing 'dataset' section"):
            get_datamodule(OmegaConf.create({"paths": {}}))

    def test_missing_dataset_name_raises(self):
        """Test that missing dataset.name raises ValueError."""
        cfg = OmegaConf.create({"dataset": {"modality": "h5"}})
        with pytest.raises(ValueError, match="Missing 'dataset.name'"):
            get_datamodule(cfg)

    def test_missing_dataset_modality_raises(self):
        """Test that missing dataset.modality raises ValueError."""
        cfg = OmegaConf.create({"dataset": {"name": "ddacs"}})
        with pytest.raises(ValueError, match="Missing 'dataset.modality'"):
            get_datamodule(cfg)

    def test_invalid_modality_raises(self):
        """Test that invalid modality raises ValueError."""
        cfg = OmegaConf.create(
            {
                "dataset": {
                    "name": "ddacs",
                    "modality": "nonexistent",
                    "modalities": {"h5": {}},
                },
            }
        )
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            get_datamodule(cfg)

    def test_missing_datamodule_class_raises(self):
        """Test that missing datamodule_class raises ValueError."""
        cfg = OmegaConf.create(
            {
                "dataset": {
                    "name": "ddacs",
                    "modality": "h5",
                    "modalities": {"h5": {"data_dir": "/tmp"}},
                },
            }
        )
        with pytest.raises(ValueError, match="Missing 'datamodule_class'"):
            get_datamodule(cfg)

    def test_kwargs_are_forwarded(self):
        """Test that modality kwargs reach the DataModule constructor."""
        cfg = OmegaConf.create(
            {
                "dataset": {
                    "name": "ddacs",
                    "modality": "h5",
                    "modalities": {
                        "h5": {
                            "datamodule_class": "src.data.ddacs.StreamingDDACS",
                            "data_dir": "/tmp/ddacs/h5",
                            "kwargs": {"view": "springback-minimal", "seed": 7},
                        },
                    },
                },
                "training": {"batch_size": 8, "num_workers": 0},
            }
        )
        dm = get_datamodule(cfg)
        assert isinstance(dm, StreamingDDACS)
        assert dm.view == "springback-minimal"
        assert dm.seed == 7
        assert dm.batch_size == 8
        assert str(dm.root) == "/tmp/ddacs"


class TestExportDatasets:
    """Unit tests for the export-based datasets (synthetic data)."""

    def test_split_filtering(self, synthetic_ddacs_root):
        """Split filter selects exactly the rows with that split value."""
        assert len(PointCloudDDACSDataset(synthetic_ddacs_root, split="train")) == 2
        assert len(PointCloudDDACSDataset(synthetic_ddacs_root, split="val")) == 1
        assert len(PointCloudDDACSDataset(synthetic_ddacs_root, split="test")) == 1
        assert len(PointCloudDDACSDataset(synthetic_ddacs_root)) == 4

    def test_invalid_split_raises(self, synthetic_ddacs_root):
        """Unknown split names are rejected."""
        with pytest.raises(ValueError, match="split must be one of"):
            PointCloudDDACSDataset(synthetic_ddacs_root, split="validation")

    def test_sample_contents(self, synthetic_ddacs_root):
        """Samples are dicts of tensors + parameters + sim_id."""
        ds = PointCloudDDACSDataset(synthetic_ddacs_root, split="train")
        sample = ds[0]
        assert sample["blank"].shape == (8, 3)
        assert sample["blank_thickness"].shape == (8,)
        assert sample["parameters"].shape == (len(PARAM_COLUMNS),)
        assert sample["parameters"].dtype == torch.float32
        assert sample["sim_id"] == 101
        assert "split" not in sample  # string fields are dropped

    def test_transform_applied(self, synthetic_ddacs_root):
        """A transform sees and can modify the sample dict."""

        def tag(sample):
            sample["tagged"] = True
            return sample

        ds = PointCloudDDACSDataset(synthetic_ddacs_root, split="train", transform=tag)
        assert ds[0]["tagged"] is True

    def test_datamodule_batches(self, synthetic_ddacs_root):
        """PointCloud/Image DataModules produce default-collated batches."""
        for dm_cls, key in [(PointCloudDDACS, "blank"), (ImageDDACS, "blank_z")]:
            dm = dm_cls(root=synthetic_ddacs_root, batch_size=2, num_workers=0)
            dm.setup("fit")
            batch = next(iter(dm.train_dataloader()))
            assert batch[key].shape[0] == 2
            assert batch["parameters"].shape == (2, len(PARAM_COLUMNS))


class TestGraphDataset:
    """Unit tests for the graph dataset and its PyG batching."""

    def test_split_and_shapes(self, synthetic_ddacs_root):
        """Time-first arrays become node-first; edge indices are long."""
        ds = GraphDDACSDataset(synthetic_ddacs_root, split="train")
        assert len(ds) == 2
        data = ds[0]["data"]
        assert data.pos.shape == (5, 2, 3)  # (N, T, 3), node-first
        assert data.edge_index.dtype == torch.long
        assert data.norm_offset.shape == (1, 3)  # graph-level: leading axis

    def test_batching_offsets_each_subgraph(self, synthetic_ddacs_root):
        """Blank and tool edge indices are offset by their own node counts."""
        ds = GraphDDACSDataset(synthetic_ddacs_root, split="train")
        batch = graph_collate_fn([ds[0], ds[1]])
        data = batch["data"]
        assert data.num_graphs == 2
        assert data.pos.shape[0] == 10  # 2 x 5 blank nodes
        assert data.edge_index.max() < data.pos.size(0)
        assert data.binder_edge_index.max() < data.binder_pos.size(0)
        assert data.norm_offset.shape == (2, 3)
        assert batch["parameters"].shape == (2, len(PARAM_COLUMNS))

    def test_datamodule(self, synthetic_ddacs_root):
        """GraphDDACS wires the PyG collate automatically."""
        dm = GraphDDACS(root=synthetic_ddacs_root, batch_size=2, num_workers=0)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["data"].num_graphs == 2


@pytest.mark.slow
@requires_ddacs_data
class TestRealData:
    """Integration: one batch from every modality of the real dataset."""

    def test_pointcloud(self, ddacs_root):
        """Point-cloud modality yields correctly shaped batches."""
        dm = PointCloudDDACS(root=ddacs_root, batch_size=2, num_workers=0)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["blank"].shape[1:] == (4, 4096, 3)
        assert batch["parameters"].shape == (2, len(PARAM_COLUMNS))

    def test_image(self, ddacs_root):
        """Image modality yields batches."""
        dm = ImageDDACS(root=ddacs_root, batch_size=2, num_workers=0)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        assert batch["blank_z"].shape[0] == 2

    def test_graph(self, ddacs_root):
        """Graph modality batches with valid edge offsets."""
        dm = GraphDDACS(root=ddacs_root, batch_size=2, num_workers=0)
        dm.setup("fit")
        data = next(iter(dm.train_dataloader()))["data"]
        assert data.num_graphs == 2
        assert data.edge_index.max() < data.pos.size(0)
        assert data.die_edge_index.max() < data.die_pos.size(0)

    def test_streaming(self, ddacs_root):
        """Streaming modality yields batches straight from the h5 zips."""
        dm = StreamingDDACS(
            root=ddacs_root, batch_size=2, num_workers=0, view="springback-minimal"
        )
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        assert all(isinstance(v, torch.Tensor) for v in batch.values())
        assert next(iter(batch.values())).shape[0] == 2
