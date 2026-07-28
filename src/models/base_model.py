"""Base model class for surrogate models.

All student implementations MUST inherit from BaseSurrogateModel.
This ensures consistent interface for training, evaluation, and comparison.

Students implement:
- __init__: Initialize model architecture
- forward: Define forward pass
- configure_optimizers: Define optimizer and scheduler

Everything else is handled automatically for fair comparison.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from src.metrics.metric_collection import MetricCollection


class BaseSurrogateModel(pl.LightningModule, ABC):
    """Abstract base class for surrogate models predicting springback in deep drawing.

    Required implementations:
        __init__: Initialize your model architecture
        forward: Define forward pass
        configure_optimizers: Define optimizer and learning rate scheduler

    Optional overrides for custom behavior:
        preprocess_data: Transform base dataloader output to model-specific
            modality (point clouds, voxels, graphs, images)
        postprocess_predictions: Transform predictions (e.g., denormalization)
        compute_loss: Use custom loss function
        training_step/validation_step/test_step: Custom training logic

    The preprocess_data() method enables different models to use the same base
    dataloader while working with different data representations.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.

        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # Metrics will be set up externally via setup_metrics()
        self.train_metrics: Optional[MetricCollection] = None
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None

        # Store predictions and targets for evaluation analysis
        self.test_predictions = []
        self.test_targets = []

    def setup_metrics(self, metrics: MetricCollection):
        """Setup metrics for training/validation/testing.

        Args:
            metrics: MetricCollection instance with all evaluation metrics

        Note: This is called automatically by the training script.
              Students don't need to call this.
        """
        self.train_metrics = metrics
        self.val_metrics = metrics
        self.test_metrics = metrics

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            predictions: Output tensor of shape (batch_size, output_dim)
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Returns:
            Optimizer or tuple of (optimizer, scheduler)
        """
        pass

    def preprocess_data(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform dataloader output to model-specific format.

        Override this method to adapt the base dataset to your model's required
        modality (point clouds, voxels, graphs, images, etc.). This allows all
        students to use the same base dataloader while supporting different
        model architectures.

        Examples:
            Point cloud model: Convert to (N, 3) point coordinates
            Voxel model: Convert to (C, D, H, W) voxel grid
            Graph model: Convert to edge indices and node features
            Image model: Convert to (C, H, W) image tensor

        Args:
            batch: Batch from dataloader (default: tuple of (inputs, targets))

        Returns:
            Tuple of (inputs, targets) in model-specific format
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        else:
            raise ValueError(
                "Batch should be a tuple/list of (inputs, targets). "
                "Override preprocess_data() for custom preprocessing."
            )

    def postprocess_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Postprocess model predictions.

        Override this method for custom postprocessing (e.g., denormalization).

        Args:
            predictions: Raw model predictions

        Returns:
            Processed predictions
        """
        return predictions

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between predictions and targets.

        Override this method for custom loss functions.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Loss value
        """
        # Default: MSE loss
        return nn.functional.mse_loss(predictions, targets)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch from training dataloader
            batch_idx: Index of the batch

        Returns:
            Loss value
        """
        inputs, targets = self.preprocess_data(batch)
        predictions = self(inputs)
        predictions = self.postprocess_predictions(predictions)
        loss = self.compute_loss(predictions, targets)
        bs = targets.shape[0]

        # Log training loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )

        # Log metrics if available
        if self.train_metrics is not None:
            metrics = self.train_metrics(predictions, targets)
            self.log_dict(
                {f"train/{k}": v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch from validation dataloader
            batch_idx: Index of the batch

        Returns:
            Loss value
        """
        inputs, targets = self.preprocess_data(batch)
        predictions = self(inputs)
        predictions = self.postprocess_predictions(predictions)
        loss = self.compute_loss(predictions, targets)
        bs = targets.shape[0]

        # Log validation loss
        self.log(
            "val/loss",
            loss.detach().clone(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )

        # Log metrics if available
        if self.val_metrics is not None:
            metrics = self.val_metrics(predictions, targets)
            self.log_dict(
                {f"val/{k}": v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Batch from test dataloader
            batch_idx: Index of the batch

        Returns:
            Loss value
        """
        inputs, targets = self.preprocess_data(batch)
        predictions = self(inputs)
        predictions = self.postprocess_predictions(predictions)
        loss = self.compute_loss(predictions, targets)
        bs = targets.shape[0]

        # Store predictions and targets for final evaluation
        self.test_predictions.append(predictions.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=bs)

        # Log metrics if available
        if self.test_metrics is not None:
            metrics = self.test_metrics(predictions, targets)
            self.log_dict(
                {f"test/{k}": v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )

        return loss

    def on_test_epoch_end(self):
        """Called at the end of test epoch.

        Concatenates all predictions and targets for final evaluation.
        """
        if self.test_predictions:
            self.all_test_predictions = torch.cat(self.test_predictions, dim=0)
            self.all_test_targets = torch.cat(self.test_targets, dim=0)

            # Clear batch storage
            self.test_predictions = []
            self.test_targets = []

    def get_test_results(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all test predictions and targets.

        Returns:
            Tuple of (predictions, targets)

        Raises:
            ValueError: if it has no attribute all_test_predictions.
        """
        if hasattr(self, "all_test_predictions"):
            return self.all_test_predictions, self.all_test_targets
        else:
            raise ValueError

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step.

        Args:
            batch: Batch from dataloader
            batch_idx: Index of the batch

        Returns:
            Predictions
        """
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch

        predictions = self(inputs)
        return self.postprocess_predictions(predictions)
