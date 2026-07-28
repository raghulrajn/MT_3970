"""3D point cloud metrics for springback prediction.

Uses point-cloud-utils library for reliable metric calculations.
"""

import numpy as np
import point_cloud_utils as pcu
import torch

from .base_metric import BaseMetric


class EPE3D(BaseMetric):
    """End Point Error in 3D using point-cloud-utils."""

    def __init__(self):
        super().__init__("epe3d")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean Euclidean distance between points.

        Args:
            predictions: Predicted 3D points, shape (..., 3).
            targets: Ground truth 3D points, shape (..., 3).

        Returns:
            Mean EPE3D value.
        """
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 3)
        target_np = targets.detach().cpu().numpy().reshape(-1, 3)

        distances = pcu.pairwise_distances(pred_np, target_np)
        epe = np.mean(np.diag(distances))

        return torch.tensor(epe, dtype=predictions.dtype, device=predictions.device)


class Chamfer3D(BaseMetric):
    """Chamfer distance for 3D point clouds using point-cloud-utils."""

    def __init__(self):
        super().__init__("chamfer3d")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance between point clouds.

        Args:
            predictions: Predicted 3D points, shape (..., 3).
            targets: Ground truth 3D points, shape (..., 3).

        Returns:
            Chamfer distance.
        """
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 3)
        target_np = targets.detach().cpu().numpy().reshape(-1, 3)

        chamfer_dist = pcu.chamfer_distance(pred_np, target_np)

        return torch.tensor(
            chamfer_dist, dtype=predictions.dtype, device=predictions.device
        )


class HausdorffDistance(BaseMetric):
    """Hausdorff distance for 3D point clouds using point-cloud-utils."""

    def __init__(self):
        super().__init__("hausdorff")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff distance between point clouds.

        Args:
            predictions: Predicted 3D points, shape (..., 3).
            targets: Ground truth 3D points, shape (..., 3).

        Returns:
            Hausdorff distance.
        """
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 3)
        target_np = targets.detach().cpu().numpy().reshape(-1, 3)

        hausdorff_dist = pcu.hausdorff_distance(pred_np, target_np)

        return torch.tensor(
            hausdorff_dist, dtype=predictions.dtype, device=predictions.device
        )


class Acc3DStrict(BaseMetric):
    """3D accuracy with strict threshold."""

    def __init__(self, threshold: float = 0.05):
        super().__init__("acc3d_strict")
        self.threshold = threshold

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute percentage of points within threshold distance.

        Args:
            predictions: Predicted 3D points, shape (..., 3).
            targets: Ground truth 3D points, shape (..., 3).

        Returns:
            Accuracy percentage.
        """
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 3)
        target_np = targets.detach().cpu().numpy().reshape(-1, 3)

        distances = np.linalg.norm(pred_np - target_np, axis=1)
        correct = (distances < self.threshold).astype(float)
        accuracy = np.mean(correct) * 100

        return torch.tensor(
            accuracy, dtype=predictions.dtype, device=predictions.device
        )


class MaxError3D(BaseMetric):
    """Maximum 3D error."""

    def __init__(self):
        super().__init__("max_error_3d")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute maximum Euclidean distance.

        Args:
            predictions: Predicted 3D points, shape (..., 3).
            targets: Ground truth 3D points, shape (..., 3).

        Returns:
            Maximum distance.
        """
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 3)
        target_np = targets.detach().cpu().numpy().reshape(-1, 3)

        distances = np.linalg.norm(pred_np - target_np, axis=1)
        max_dist = np.max(distances)

        return torch.tensor(
            max_dist, dtype=predictions.dtype, device=predictions.device
        )
