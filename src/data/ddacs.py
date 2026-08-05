"""DDACS dataset modules built on the ddacs 3.x package.

Based on the DDACS dataset: https://doi.org/10.18419/DARUS-4801

The data lives under ``<root>/`` (default ``$DATA_DIR/datasets/ddacs``) in
the formats produced by the ddacs 3.x export tools::

    /mnt/data/datasets/ddacs/
    ├── h5/                       # Raw HDF5 zips (streamed, no export needed)
    ├── pointcloud/               # export_to_numpy dir (.npy stacks)
    ├── image/                    # export_to_numpy dir (.npy stacks)
    ├── graph/                    # one <sim_id>.npz per simulation
    ├── metadata.json             # Croissant manifest (view definitions)
    └── process_parameters.csv    # parameters + train/val/test split

- ``pointcloud/`` / ``image/`` are read lazily (memmap) via
  :func:`ddacs.streaming.load_export` - map-style, fast random access.
- ``graph/`` holds variable-size meshes/graphs, one ``.npz`` per sim.
- ``h5/`` is streamed sample-by-sample via
  :class:`ddacs.pytorch.DDACSDataset` (IterableDataset): any view, no
  export step, at the cost of sequential access.

Train/val/test splits come from the ``split`` column of
``process_parameters.csv`` (also stored per-record in the exports).
"""

import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.data.base_datamodule import BaseDataModule

logger = logging.getLogger(__name__)

#: Numeric process parameters (columns of process_parameters.csv) that make up
#: the ``parameters`` tensor of every sample, in this order.
# PARAM_COLUMNS = [
#     "curvature_radius",
#     "bottom_radius",
#     "wall_angle",
#     "material_scaling_factor",
#     "sheet_metal_thickness",
#     "friction_coefficient",
#     "blankholder_force",
# ]

PARAM_COLUMNS = [
    "curvature_radius",
    "material_scaling_factor",
    "sheet_metal_thickness",
    "friction_coefficient",
    "blankholder_force",
]

VALID_SPLITS = ("train", "val", "test")

#: Arrays stored time-first (T, N, ...) in the .npz graph exports; converted
#: to node-first (N, T, ...) so PyG batching concatenates along the node axis.
_TIME_FIRST_KEYS = (
    "pos",
    "pos_delta",
    "op20_pos",
    "contact_direction",
    "contact_distance",
    "edge_rel_pos",
    "binder_pos",
    "die_pos",
    "punch_pos",
)

#: Per-simulation (graph-level) vectors; get a leading batch axis so a
#: batch of B graphs yields shape (B, ...).
_GRAPH_LEVEL_KEYS = ("norm_offset", "norm_scale", "die_height", "stamping_speed")


def load_process_parameters(root: Path) -> pd.DataFrame:
    """Load process_parameters.csv indexed by simulation id.

    Args:
        root: DDACS dataset root (contains process_parameters.csv).

    Returns:
        DataFrame indexed by sim id (the ``index`` column of the CSV).

    Raises:
        FileNotFoundError: If the CSV is missing under root.
    """
    csv_path = Path(root) / "process_parameters.csv"
    # csv_path = Path("/home/RUS_CIP/st189432/master-thesis-template-master/process_parameters_1.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"process_parameters.csv not found under {root}. "
            f"Is DATA_DIR set correctly?"
        )
    return pd.read_csv(csv_path).set_index("index")


def _record_to_tensors(record: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert a ddacs record (dict of numpy values) to torch tensors.

    Numeric arrays and scalars become tensors; string-valued entries
    (e.g. ``split``, ``geometry``) are dropped - they are bookkeeping,
    not model input.
    """
    sample: Dict[str, torch.Tensor] = {}
    for key, value in record.items():
        arr = np.asarray(value)
        if arr.dtype.kind in ("U", "S", "O"):
            continue
        # copy: exports are read-only memmaps, torch needs writable memory
        sample[key] = torch.from_numpy(np.array(arr))
    return sample


def _split_is(split: str, row: pd.Series) -> bool:
    """Module-level split predicate (picklable for DataLoader workers)."""
    return row["split"] == split


class ExportDDACSDataset(Dataset):
    """Map-style dataset over a ddacs ``export_to_numpy`` directory.

    Samples are dicts of tensors keyed by the exported field names, plus
    ``parameters`` (the :data:`PARAM_COLUMNS` vector) and ``sim_id``.
    All fields have fixed shapes, so the default collate works - no
    custom collate_fn required.

    Args:
        root: DDACS dataset root.
        subdir: Export directory below root (``pointcloud`` or ``image``).
        split: Optional split filter ('train', 'val', 'test'); None = all.
        transform: Optional callable applied to each sample dict.
        fields: Optional subset of export fields to load (default: all).
    """

    def __init__(
        self,
        root: str,
        subdir: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        fields: Optional[List[str]] = None,
    ):
        from ddacs.streaming import load_export

        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        self.root = Path(root)
        self.transform = transform
        export_dir = self.root / subdir
        self.export = load_export(export_dir, fields=fields)
        self.sim_ids = np.load(export_dir / "sim_ids.npy")

        params = load_process_parameters(self.root)
        params = params[~params.index.duplicated(keep="first")]

        # Keep only simulations explicitly present in process_parameters.csv.
        # This allows using a reduced testing CSV (subset of sim ids) without
        # iterating over unrelated export records.
        keep_mask = np.isin(self.sim_ids, params.index.to_numpy())

        if split is not None:
            if "split" in params.columns:
                split_ids = params.index[params["split"] == split].to_numpy()
                keep_mask &= np.isin(self.sim_ids, split_ids)
            else:
                splits = np.load(export_dir / "split.npy")
                keep_mask &= splits == split

        self.indices = np.nonzero(keep_mask)[0]
        self._selected_sim_ids = self.sim_ids[self.indices]

        param_subset = params.reindex(index=self._selected_sim_ids)[PARAM_COLUMNS]
        self._parameters = torch.from_numpy(param_subset.to_numpy(dtype=np.float32))

        logger.info(
            f"{type(self).__name__}: {len(self.indices)} samples "
            f"(split={split}) from {export_dir}"
        )

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one sample as a dict of tensors + parameters + sim_id."""
        pos = int(self.indices[idx])
        sample: Dict[str, Any] = _record_to_tensors(self.export[pos])
        sample["parameters"] = self._parameters[idx]
        PARAM_MIN = torch.tensor([30.0, 0.9, 0.05, 0.95, 100000.0], dtype=torch.float32)
        PARAM_MAX = torch.tensor([150.0, 1.1, 0.15, 1.00, 500000.0], dtype=torch.float32
        )
        sample["parameters"] = (sample["parameters"] - PARAM_MIN) / (PARAM_MAX - PARAM_MIN)
        sample["sim_id"] = int(self._selected_sim_ids[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class PointCloudDDACSDataset(ExportDDACSDataset):
    """Point clouds of blank/die/punch/binder with stress/strain/thickness."""

    def __init__(self, root: str, split: Optional[str] = None, **kwargs):
        super().__init__(root, "pointcloud", split=split, fields=["blank", "die", "punch", "binder"], **kwargs)


class ImageDDACSDataset(ExportDDACSDataset):
    """Height-map / displacement image representation."""

    def __init__(self, root: str, split: Optional[str] = None, **kwargs):
        super().__init__(root, "image", split=split, **kwargs)


class DDACSData(Data):
    """PyG Data with correct batching for the DDACS multi-graph records.

    One record holds four graphs (blank mesh + die/punch/binder tools),
    each with its own ``*_edge_index``. PyG's default batching would offset
    every edge_index by the *blank* node count; this class offsets each
    tool's edge_index by that tool's own node count.
    """

    def __inc__(self, key, value, *args, **kwargs):
        """Return the batch-offset increment for the given attribute."""
        if key == "edge_index":
            return self.pos.size(0)
        for tool in ("binder", "die", "punch"):
            if key == f"{tool}_edge_index":
                return getattr(self, f"{tool}_pos").size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDDACSDataset(Dataset):
    """Map-style dataset over the per-simulation graph ``.npz`` exports.

    Each sample is a dict with a :class:`DDACSData` graph object (blank
    mesh ``pos``/``edge_index``/``node_type``, the die/punch/binder tool
    subgraphs, op20 masks and contact features), the ``parameters`` vector
    and the ``sim_id``. Time-first arrays (T, N, ...) are stored node-first
    (N, T, ...) so PyG batching works. Use :func:`graph_collate_fn`.

    Args:
        root: DDACS dataset root (graph files under ``root/graph``).
        split: Optional split filter; None = all simulations in the CSV.
        transform: Optional callable applied to each sample dict.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        self.root = Path(root)
        self.graph_dir = self.root / "graph"
        self.transform = transform

        params = load_process_parameters(self.root)
        if split is not None:
            params = params[params["split"] == split]
        self.sim_ids = params.index.to_numpy()
        self._parameters = torch.from_numpy(
            params[PARAM_COLUMNS].to_numpy(dtype=np.float32)
        )

        logger.info(
            f"GraphDDACSDataset: {len(self.sim_ids)} samples "
            f"(split={split}) from {self.graph_dir}"
        )

    def __len__(self) -> int:
        """Return the number of simulations in the selected split."""
        return len(self.sim_ids)

    def __getitem__(self, idx: int) -> Dict:
        """Return one sample: DDACSData graph + parameters + sim_id."""
        sim_id = int(self.sim_ids[idx])
        with np.load(self.graph_dir / f"{sim_id}.npz") as npz:
            attrs = _record_to_tensors(dict(npz))

        for key in list(attrs):
            if key in _TIME_FIRST_KEYS:
                # (T, N, ...) -> (N, T, ...): PyG concatenates along nodes
                attrs[key] = attrs[key].movedim(0, 1).contiguous()
            elif key in _GRAPH_LEVEL_KEYS:
                # leading batch axis: a batch of B graphs yields (B, ...)
                attrs[key] = attrs[key].unsqueeze(0)
            elif key.endswith("edge_index"):
                attrs[key] = attrs[key].long()

        data = DDACSData(num_nodes=attrs["pos"].size(0), **attrs)
        sample = {
            "data": data,
            "parameters": self._parameters[idx],
            "sim_id": sim_id,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def graph_collate_fn(batch: List[Dict]) -> Dict:
    """Collate graph samples: PyG Batch for graphs, stacked tensors otherwise.

    Args:
        batch: List of sample dicts from GraphDDACSDataset.

    Returns:
        Dict with ``data`` (torch_geometric Batch), ``parameters``
        (batch, len(PARAM_COLUMNS)) and ``sim_id`` (batch,).
    """
    return {
        "data": Batch.from_data_list([s["data"] for s in batch]),
        "parameters": torch.stack([s["parameters"] for s in batch]),
        "sim_id": torch.tensor([s["sim_id"] for s in batch]),
    }


class PointCloudDDACS(BaseDataModule):
    """DataModule for the point-cloud modality (default collate)."""

    def _create_dataset(self, split: str) -> Dataset:
        return PointCloudDDACSDataset(self.root, split=split, transform=self.transform)


class ImageDDACS(BaseDataModule):
    """DataModule for the image modality (default collate)."""

    def _create_dataset(self, split: str) -> Dataset:
        return ImageDDACSDataset(self.root, split=split, transform=self.transform)


class GraphDDACS(BaseDataModule):
    """DataModule for the graph/mesh modality (PyG Batch collate)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("collate_fn", graph_collate_fn)
        super().__init__(*args, **kwargs)

    def _create_dataset(self, split: str) -> Dataset:
        return GraphDDACSDataset(self.root, split=split, transform=self.transform)


class StreamingDDACS(pl.LightningDataModule):
    """Streams samples directly from the raw HDF5 archives (ddacs 3.x).

    No export step needed: any registered view can be streamed straight
    from ``<root>/h5/*.zip`` via :class:`ddacs.pytorch.DDACSDataset`
    (an ``IterableDataset``). Use this to work with fields that are not in
    the pre-exported modalities, at the cost of sequential access.

    Notes:
        - Batches are dicts of tensors keyed by the view's field aliases
          (the default collate converts the streamed numpy arrays).
        - Shuffling is per worker-shard; ``StreamingShuffleCallback``
          (registered automatically by ``setup_callbacks``) reshuffles
          each epoch via ``set_epoch``.
        - ``__len__`` is unavailable (IterableDataset) - progress bars
          show no total and ``limit_*_batches`` fractions won't work.

    Args:
        root: DDACS dataset root (zips under ``root/h5``).
        batch_size: Batch size for DataLoaders.
        num_workers: DataLoader workers (sharding is handled by ddacs).
        view: View name to stream, e.g. ``graph-blank-tools``,
            ``pointcloud-blank-tools``, ``image-blank-tools`` (registered
            by this module) or any RecordSet in metadata.json
            (e.g. ``springback-minimal``).
        seed: Base seed for the per-shard shuffle.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        view: str = "graph-blank-tools",
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.view = view
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Register the view recipes and create per-split streams."""
        import ddacs
        from ddacs import views
        from ddacs.pytorch import DDACSDataset

        croissant = ddacs.load(data_dir=self.root)
        for recipe in (
            views.pointcloud_blank_tools,
            views.image_blank_tools,
            views.graph_blank_tools,
        ):
            recipe(croissant)

        def make(split: str, shuffle: bool) -> DDACSDataset:
            return DDACSDataset(
                self.view,
                data_dir=self.root,
                dataset=croissant,
                where=partial(_split_is, split),
                shuffle=shuffle,
                seed=self.seed,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = make("train", shuffle=True)
            self.val_dataset = make("val", shuffle=False)
        if stage == "test" or stage is None:
            self.test_dataset = make("test", shuffle=False)

    def _dataloader(self, dataset) -> DataLoader:
        if dataset is None:
            raise RuntimeError(
                "You must call .setup() before accessing the dataloader."
            )
        # No shuffle/sampler args: the IterableDataset shards and shuffles
        # internally across workers and DDP ranks.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the streaming DataLoader for the train split."""
        return self._dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        """Return the streaming DataLoader for the val split."""
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Return the streaming DataLoader for the test split."""
        return self._dataloader(self.test_dataset)
