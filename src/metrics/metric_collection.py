"""Metric collection for managing multiple metrics.

Allows easy computation and tracking of multiple metrics at once.
"""

import logging
from typing import Dict, List

import torch

from .base_metric import BaseMetric

log = logging.getLogger(__name__)


class MetricCollection:
    """Collection of metrics for easy management.

    Example usage:
        >>> metrics = MetricCollection.create_default()
        >>> results = metrics(predictions, targets)
        >>> print(results)
        {"mse": ..., "rmse": ..., "mae": ...}

        # Filter to specific metrics
        >>> filtered = metrics.select(["mse", "rmse"])
        >>> results = filtered(predictions, targets)
        {"mse": ..., "rmse": ...}
    """

    def __init__(self, metrics: List[BaseMetric]):
        """Initialize metric collection.

        Args:
            metrics: List of metric instances
        """
        self.metrics = {metric.name: metric for metric in metrics}

    @classmethod
    def create_default(cls) -> "MetricCollection":
        """Create a MetricCollection with all available metrics.

        Returns:
            MetricCollection containing all registered metrics.

        Note:
            To add a new metric:
            1. Create the metric class in regression_metrics.py or point_cloud_metrics.py
            2. Add it to the appropriate section below
            3. The metric will be available via select() using its name
        """
        from .point_cloud_metrics import (
            EPE3D,
            Acc3DStrict,
            Chamfer3D,
            HausdorffDistance,
            MaxError3D,
        )
        from .regression_metrics import MAE, MAPE, MSE, RMSE, R2Score

        # =====================================================================
        # ADD NEW METRICS HERE
        # =====================================================================
        # Regression metrics (for scalar predictions)
        regression_metrics = [
            MSE(),
            RMSE(),
            MAE(),
            R2Score(),
            MAPE(),
        ]

        # Point cloud metrics (for 3D geometry)
        point_cloud_metrics = [
            EPE3D(),
            Acc3DStrict(),
            Chamfer3D(),
            HausdorffDistance(),
            MaxError3D(),
        ]
        # =====================================================================

        all_metrics = regression_metrics + point_cloud_metrics
        return cls(all_metrics)

    def select(self, names: List[str]) -> "MetricCollection":
        """Create a new MetricCollection with only the specified metrics.

        Args:
            names: List of metric names to include (case-insensitive).

        Returns:
            New MetricCollection with only the selected metrics.

        Raises:
            ValueError: If none of the specified metrics are found.

        Example:
            >>> all_metrics = MetricCollection.create_default()
            >>> selected = all_metrics.select(["mse", "rmse", "mae"])
        """
        names_lower = [n.lower() for n in names]
        selected = []
        found_names = []

        for name, metric in self.metrics.items():
            if name.lower() in names_lower:
                selected.append(metric)
                found_names.append(name.lower())

        # Warn about metrics that weren't found
        not_found = [n for n in names_lower if n not in found_names]
        if not_found:
            log.warning(
                f"Metrics not found: {not_found}. "
                f"Available: {list(self.metrics.keys())}"
            )

        if not selected:
            raise ValueError(
                f"None of the specified metrics {names} were found. "
                f"Available: {list(self.metrics.keys())}"
            )

        return MetricCollection(selected)

    def available_metrics(self) -> List[str]:
        """Get list of available metric names in this collection.

        Returns:
            List of metric names.
        """
        return list(self.metrics.keys())

    def add_metric(self, metric: BaseMetric):
        """Add a metric to the collection.

        Args:
            metric: Metric instance to add.

        Note:
            If a metric with the same name exists, it will be overwritten.
        """
        if metric.name in self.metrics:
            log.warning(f"Overwriting existing metric '{metric.name}'")
        self.metrics[metric.name] = metric

    def remove_metric(self, name: str):
        """Remove a metric from the collection.

        Args:
            name: Name of the metric to remove (case-sensitive).

        Raises:
            KeyError: If metric with given name does not exist.
        """
        if name not in self.metrics:
            raise KeyError(
                f"Metric '{name}' not found. " f"Available: {list(self.metrics.keys())}"
            )
        del self.metrics[name]

    def reset(self):
        """Reset all metrics in the collection."""
        for metric in self.metrics.values():
            metric.reset()

    def compute(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all metrics in the collection.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary mapping metric names to values
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute(predictions, targets)
            except Exception as e:
                log.warning(f"Failed to compute metric '{name}': {e}")
                results[name] = torch.tensor(float("nan"))

        return results

    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Callable interface for computing all metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary mapping metric names to values
        """
        return self.compute(predictions, targets)

    def __repr__(self) -> str:
        """String representation of the collection."""
        metric_names = ", ".join(self.metrics.keys())
        return f"MetricCollection({metric_names})"
