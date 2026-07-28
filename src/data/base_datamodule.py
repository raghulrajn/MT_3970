"""Base DataModule for standardized train/val/test data loading.

Provides a reusable base class that handles common DataModule boilerplate:
- DataLoader creation with configurable collate functions
- Setup method with train/val/test split handling
- Runtime checks for proper initialization

Subclasses only need to implement `_create_dataset(split)` to specify
which Dataset class to use.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(pl.LightningDataModule, ABC):
    """Base DataModule with common train/val/test dataloader logic.

    Subclasses must implement `_create_dataset(split)` to return the
    appropriate Dataset for each split.

    Args:
        root: Root directory of the dataset.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for DataLoaders.
        transform: Optional transform to apply to samples.
        collate_fn: Optional custom collate function for batching.
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.collate_fn = collate_fn

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @abstractmethod
    def _create_dataset(self, split: str) -> Dataset:
        """Create a dataset for the given split.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            Dataset instance for the specified split.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test.

        Args:
            stage: One of 'fit', 'test', or None (for both).
        """
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._create_dataset("test")

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training set.

        Returns:
            DataLoader with shuffled training data.

        Raises:
            RuntimeError: If setup() was not called before.
        """
        if self.train_dataset is None:
            raise RuntimeError(
                "You must call .setup() before accessing the dataloader."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation set.

        Returns:
            DataLoader with validation data.

        Raises:
            RuntimeError: If setup() was not called before.
        """
        if self.val_dataset is None:
            raise RuntimeError(
                "You must call .setup() before accessing the dataloader."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test set.

        Returns:
            DataLoader with test data.

        Raises:
            RuntimeError: If setup() was not called before.
        """
        if self.test_dataset is None:
            raise RuntimeError(
                "You must call .setup() before accessing the dataloader."
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
