"""Standard regression metrics for model evaluation.

Includes MSE, RMSE, MAE, R², and MAPE.
"""

import torch
import torch.nn.functional as F

from .base_metric import BaseMetric


class MSE(BaseMetric):
    """Mean Squared Error."""

    def __init__(self):
        super().__init__("mse")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            MSE value
        """
        return F.mse_loss(predictions, targets)


class RMSE(BaseMetric):
    """Root Mean Squared Error."""

    def __init__(self):
        super().__init__("rmse")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute RMSE.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            RMSE value
        """
        mse = F.mse_loss(predictions, targets)
        return torch.sqrt(mse)


class MAE(BaseMetric):
    """Mean Absolute Error."""

    def __init__(self):
        super().__init__("mae")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAE.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            MAE value
        """
        return F.l1_loss(predictions, targets)


class R2Score(BaseMetric):
    """Coefficient of Determination (R² Score)."""

    def __init__(self):
        super().__init__("r2")

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute R² score.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            R² value
        """
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

        # Avoid division by zero
        if ss_tot == 0:
            return torch.tensor(0.0, device=predictions.device)

        r2 = 1 - (ss_res / ss_tot)
        return r2


class MAPE(BaseMetric):
    """Mean Absolute Percentage Error."""

    def __init__(self, epsilon: float = 1e-8):
        """Initialize MAPE metric.

        Args:
            epsilon: Small value to avoid division by zero
        """
        super().__init__("mape")
        self.epsilon = epsilon

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAPE.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            MAPE value (in percentage)
        """
        # Add epsilon to avoid division by zero
        mape = (
            torch.mean(torch.abs((targets - predictions) / (targets + self.epsilon)))
            * 100
        )
        return mape
