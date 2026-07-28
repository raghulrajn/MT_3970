"""HSH DataModules for different modalities.

Dataset structure:
    /mnt/data/datasets/hsh/
    ├── pointcloud/             # Point clouds as PyG Data objects
    │   ├── blanks.pt          # List[Data] with x (2048, 3)
    │   ├── dies.pt            # List[Data] with x (2048, 3)
    │   ├── punches.pt         # List[Data] with x (2048, 3)
    │   └── binders.pt         # List[Data] with x (2048, 3)
    ├── graph/                  # Full meshes as PyG Data objects
    │   ├── blanks.pt          # List[Data] with x, face, edge_index, face_attr
    │   ├── dies.pt            # List[Data] with x, face, edge_index
    │   ├── punches.pt         # List[Data]
    │   └── binders.pt         # List[Data]
    └── metadata.csv           # Parameters and split info
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.base_datamodule import BaseDataModule


class PointCloudHSHDataset(Dataset):
    """PyTorch Dataset for HSH point clouds using PyG Data objects.

    Loads all point clouds into memory at initialization for fast access.
    Each point cloud is a PyG Data object with x: (2048, 3) point positions.

    Args:
        root: Root directory of the dataset.
        split: One of 'train', 'val', 'test', or None for all data.
        transform: Optional transform to apply to Data objects.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.metadata = pd.read_csv(self.root / "metadata.csv")
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(
                drop=True
            )

        pointcloud_dir = self.root / "pointcloud"
        self.blanks = torch.load(pointcloud_dir / "blanks.pt", weights_only=False)
        self.dies = torch.load(pointcloud_dir / "dies.pt", weights_only=False)
        self.punches = torch.load(pointcloud_dir / "punches.pt", weights_only=False)
        self.binders = torch.load(pointcloud_dir / "binders.pt", weights_only=False)

        self.param_columns = [
            "fillet_radius",
            "corner_radius",
            "bevel_angle",
            "clearance",
            "drawing_depth",
            "blankholder_force",
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dict containing:
                - blank: PyG Data object with x (2048, 3)
                - die: PyG Data object with x (2048, 3)
                - punch: PyG Data object with x (2048, 3)
                - binder: PyG Data object with x (2048, 3)
                - parameters: Tensor of shape (6,) with process parameters
                - geometry_index: Index into dies/punches/binders arrays
                - sample_index: Original sample index
        """
        row = self.metadata.iloc[idx]

        sample_index = int(row["index"])
        geometry_index = int(row["geometry_index"])

        # Clone to avoid modifying cached data
        blank = self.blanks[sample_index].clone()
        die = self.dies[geometry_index].clone()
        punch = self.punches[geometry_index].clone()
        binder = self.binders[geometry_index].clone()

        if self.transform is not None:
            blank = self.transform(blank)
            die = self.transform(die)
            punch = self.transform(punch)
            binder = self.transform(binder)

        params = torch.tensor(
            [row[col] for col in self.param_columns], dtype=torch.float32
        )

        return {
            "blank": blank,
            "die": die,
            "punch": punch,
            "binder": binder,
            "parameters": params,
            "geometry_index": geometry_index,
            "sample_index": sample_index,
        }


def pointcloud_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for PyG point cloud data.

    Batches PyG Data objects using Batch.from_data_list() for proper graph batching.

    Args:
        batch: List of sample dicts from PointCloudHSHDataset.

    Returns:
        Collated batch dict with batched PyG Data objects.
    """
    from torch_geometric.data import Batch

    return {
        "blank": Batch.from_data_list([s["blank"] for s in batch]),
        "die": Batch.from_data_list([s["die"] for s in batch]),
        "punch": Batch.from_data_list([s["punch"] for s in batch]),
        "binder": Batch.from_data_list([s["binder"] for s in batch]),
        "parameters": torch.stack([s["parameters"] for s in batch]),
        "geometry_index": torch.tensor([s["geometry_index"] for s in batch]),
        "sample_index": torch.tensor([s["sample_index"] for s in batch]),
    }


class PointCloudHSH(BaseDataModule):
    """PyTorch Lightning DataModule for HSH PointCloud modality.

    Uses PyG Data objects for point clouds (x: 2048x3 positions).

    Args:
        root: Root directory of the HSH dataset.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.
        transform: Optional transform to apply to samples.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            collate_fn=pointcloud_collate_fn,
        )

    def _create_dataset(self, split: str) -> Dataset:
        """Create PointCloudHSHDataset for the given split."""
        return PointCloudHSHDataset(
            root=str(self.root), split=split, transform=self.transform
        )


class SurfaceHSHDataset(Dataset):
    """PyTorch Dataset for HSH surface (graph/mesh) data using PyTorch Geometric.

    Loads all graph data into memory at initialization.
    Each sample contains PyG Data objects for blank, die, punch, and binder.

    The Data objects contain:
        - x: vertex positions (N, 3)
        - face: quad face indices (4, F)
        - edge_index: edge connectivity (2, E)
        - face_attr: thickness per face (F, 1) - only for blanks

    Args:
        root: Root directory of the dataset.
        split: One of 'train', 'val', 'test', or None for all data.
        transform: Optional PyG transform to apply.
    """

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.metadata = pd.read_csv(self.root / "metadata.csv")
        if split is not None:
            self.metadata = self.metadata[self.metadata["split"] == split].reset_index(
                drop=True
            )

        graph_dir = self.root / "graph"
        self.blanks = torch.load(graph_dir / "blanks.pt", weights_only=False)
        self.dies = torch.load(graph_dir / "dies.pt", weights_only=False)
        self.punches = torch.load(graph_dir / "punches.pt", weights_only=False)
        self.binders = torch.load(graph_dir / "binders.pt", weights_only=False)

        self.param_columns = [
            "fillet_radius",
            "corner_radius",
            "bevel_angle",
            "clearance",
            "drawing_depth",
            "blankholder_force",
        ]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dict containing:
                - blank: PyG Data with x, edge_index, face, face_attr
                - die: PyG Data with x, edge_index, face
                - punch: PyG Data with x, edge_index, face
                - binder: PyG Data with x, edge_index, face
                - parameters: Tensor of shape (6,) with process parameters
                - geometry_index: Index into dies/punches/binders arrays
                - sample_index: Original sample index
        """
        row = self.metadata.iloc[idx]

        sample_index = int(row["index"])
        geometry_index = int(row["geometry_index"])

        # Clone to avoid modifying cached data
        blank = self.blanks[sample_index].clone()
        die = self.dies[geometry_index].clone()
        punch = self.punches[geometry_index].clone()
        binder = self.binders[geometry_index].clone()

        if self.transform is not None:
            blank = self.transform(blank)
            die = self.transform(die)
            punch = self.transform(punch)
            binder = self.transform(binder)

        params = torch.tensor(
            [row[col] for col in self.param_columns], dtype=torch.float32
        )

        return {
            "blank": blank,
            "die": die,
            "punch": punch,
            "binder": binder,
            "parameters": params,
            "geometry_index": geometry_index,
            "sample_index": sample_index,
        }


def surface_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for PyG surface data.

    Batches PyG Data objects using Batch.from_data_list() for proper graph batching.

    Args:
        batch: List of sample dicts from SurfaceHSHDataset.

    Returns:
        Collated batch dict with batched PyG Data objects.
    """
    from torch_geometric.data import Batch

    return {
        "blank": Batch.from_data_list([s["blank"] for s in batch]),
        "die": Batch.from_data_list([s["die"] for s in batch]),
        "punch": Batch.from_data_list([s["punch"] for s in batch]),
        "binder": Batch.from_data_list([s["binder"] for s in batch]),
        "parameters": torch.stack([s["parameters"] for s in batch]),
        "geometry_index": torch.tensor([s["geometry_index"] for s in batch]),
        "sample_index": torch.tensor([s["sample_index"] for s in batch]),
    }


class SurfaceHSH(BaseDataModule):
    """PyTorch Lightning DataModule for HSH Surface (Graph/Mesh) modality.

    Uses PyTorch Geometric Data objects for efficient graph neural network training.
    Batches are created using PyG's Batch.from_data_list() for proper graph batching.

    Args:
        root: Root directory of the HSH dataset.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.
        transform: Optional transform to apply to samples.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            collate_fn=surface_collate_fn,
        )

    def _create_dataset(self, split: str) -> Dataset:
        """Create SurfaceHSHDataset for the given split."""
        return SurfaceHSHDataset(
            root=str(self.root), split=split, transform=self.transform
        )
