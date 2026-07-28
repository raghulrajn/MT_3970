"""Base metric class for evaluation.

Students can extend this class to add custom metrics.
"""

from abc import ABC, abstractmethod

import torch


class BaseMetric(ABC):
    """Abstract base class for metrics.

    All metrics should inherit from this class and implement the compute method.
    """

    def __init__(self, name: str):
        """Initialize metric.

        Args:
            name: Name of the metric (e.g., "mse", "epe3d")
        """
        self.name = name
        self.reset()

    def reset(self):
        """Reset metric state (called at the start of each epoch)."""
        pass

    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the metric.

        Args:
            predictions: Model predictions of shape (batch_size, ...)
            targets: Ground truth targets of shape (batch_size, ...)

        Returns:
            Metric value as a scalar tensor
        """
        pass

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Callable interface for computing metric.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Metric value
        """
        return self.compute(predictions, targets)
