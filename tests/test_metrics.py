"""
Unit tests for metrics system.

Tests regression metrics, point cloud metrics, and metric collection.

Run with: uv run pytest tests/test_metrics.py -v
"""

import pytest
import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.metric_collection import MetricCollection
from src.metrics.regression_metrics import MAE, MAPE, MSE, RMSE, R2Score


class TestBaseMetric:
    """Test BaseMetric abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric("test")

    def test_subclass_must_implement_compute(self):
        """Test that subclass without compute raises error."""

        class IncompleteMetric(BaseMetric):
            pass

        with pytest.raises(TypeError):
            IncompleteMetric("incomplete")

    def test_subclass_with_compute_works(self):
        """Test that proper subclass can be instantiated."""

        class SimpleMetric(BaseMetric):
            def compute(self, predictions, targets):
                return torch.mean(predictions - targets)

        metric = SimpleMetric("simple")
        assert metric.name == "simple"

    def test_callable_interface(self):
        """Test that metric can be called directly."""

        class SimpleMetric(BaseMetric):
            def compute(self, predictions, targets):
                return torch.mean((predictions - targets) ** 2)

        metric = SimpleMetric("simple")
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric(preds, targets)
        assert result == 0.0


class TestMSE:
    """Test Mean Squared Error metric."""

    def test_mse_name(self):
        """Test MSE has correct name."""
        metric = MSE()
        assert metric.name == "mse"

    def test_mse_perfect_prediction(self):
        """Test MSE is 0 for perfect predictions."""
        metric = MSE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_mse_known_value(self):
        """Test MSE computes correct value."""
        metric = MSE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])
        # Errors: [-1, 0, 1], squared: [1, 0, 1], mean: 2/3

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(2 / 3)

    def test_mse_2d_tensor(self):
        """Test MSE works with 2D tensors."""
        metric = MSE()
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)


class TestRMSE:
    """Test Root Mean Squared Error metric."""

    def test_rmse_name(self):
        """Test RMSE has correct name."""
        metric = RMSE()
        assert metric.name == "rmse"

    def test_rmse_perfect_prediction(self):
        """Test RMSE is 0 for perfect predictions."""
        metric = RMSE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_rmse_is_sqrt_of_mse(self):
        """Test RMSE is square root of MSE."""
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])

        mse = MSE().compute(preds, targets)
        rmse = RMSE().compute(preds, targets)

        assert rmse.item() == pytest.approx(torch.sqrt(mse).item())


class TestMAE:
    """Test Mean Absolute Error metric."""

    def test_mae_name(self):
        """Test MAE has correct name."""
        metric = MAE()
        assert metric.name == "mae"

    def test_mae_perfect_prediction(self):
        """Test MAE is 0 for perfect predictions."""
        metric = MAE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_mae_known_value(self):
        """Test MAE computes correct value."""
        metric = MAE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])
        # Errors: [-1, 0, 1], absolute: [1, 0, 1], mean: 2/3

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(2 / 3)


class TestR2Score:
    """Test R² Score metric."""

    def test_r2_name(self):
        """Test R2Score has correct name."""
        metric = R2Score()
        assert metric.name == "r2"

    def test_r2_perfect_prediction(self):
        """Test R² is 1 for perfect predictions."""
        metric = R2Score()
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(1.0)

    def test_r2_mean_prediction(self):
        """Test R² is 0 when predicting the mean."""
        metric = R2Score()
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = torch.full_like(targets, targets.mean())

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_r2_constant_targets(self):
        """Test R² handles constant targets (ss_tot = 0)."""
        metric = R2Score()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 2.0, 2.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)


class TestMAPE:
    """Test Mean Absolute Percentage Error metric."""

    def test_mape_name(self):
        """Test MAPE has correct name."""
        metric = MAPE()
        assert metric.name == "mape"

    def test_mape_perfect_prediction(self):
        """Test MAPE is 0 for perfect predictions."""
        metric = MAPE()
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

    def test_mape_known_value(self):
        """Test MAPE computes correct value."""
        metric = MAPE()
        preds = torch.tensor([110.0, 220.0])
        targets = torch.tensor([100.0, 200.0])
        # Percentage errors: |10/100|=0.1, |20/200|=0.1, mean=0.1 -> 10%

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(10.0, rel=0.01)


class TestMetricCollection:
    """Test MetricCollection class."""

    def test_collection_initialization(self):
        """Test MetricCollection initializes with metrics."""
        metrics = MetricCollection([MSE(), RMSE(), MAE()])

        assert "mse" in metrics.metrics
        assert "rmse" in metrics.metrics
        assert "mae" in metrics.metrics

    def test_collection_compute_all(self):
        """Test MetricCollection computes all metrics."""
        metrics = MetricCollection([MSE(), RMSE(), MAE()])
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        results = metrics.compute(preds, targets)

        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert results["mse"].item() == pytest.approx(0.0)
        assert results["rmse"].item() == pytest.approx(0.0)
        assert results["mae"].item() == pytest.approx(0.0)

    def test_collection_callable(self):
        """Test MetricCollection can be called directly."""
        metrics = MetricCollection([MSE()])
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        results = metrics(preds, targets)
        assert "mse" in results

    def test_collection_add_metric(self):
        """Test adding metric to collection."""
        metrics = MetricCollection([MSE()])
        assert "mae" not in metrics.metrics

        metrics.add_metric(MAE())
        assert "mae" in metrics.metrics

    def test_collection_remove_metric(self):
        """Test removing metric from collection."""
        metrics = MetricCollection([MSE(), MAE()])
        assert "mae" in metrics.metrics

        metrics.remove_metric("mae")
        assert "mae" not in metrics.metrics

    def test_collection_remove_nonexistent(self):
        """Test removing nonexistent metric raises KeyError."""
        metrics = MetricCollection([MSE()])
        with pytest.raises(KeyError, match="not found"):
            metrics.remove_metric("nonexistent")

    def test_collection_reset(self):
        """Test resetting all metrics in collection."""
        metrics = MetricCollection([MSE(), MAE()])
        metrics.reset()  # Should not raise

    def test_collection_repr(self):
        """Test string representation of collection."""
        metrics = MetricCollection([MSE(), MAE()])
        repr_str = repr(metrics)

        assert "MetricCollection" in repr_str
        assert "mse" in repr_str
        assert "mae" in repr_str

    def test_collection_handles_metric_error(self):
        """Test collection handles metric computation errors gracefully."""

        class FailingMetric(BaseMetric):
            def compute(self, predictions, targets):
                raise ValueError("Intentional error")

        metrics = MetricCollection([MSE(), FailingMetric("failing")])
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])

        results = metrics.compute(preds, targets)

        assert "mse" in results
        assert results["mse"].item() == pytest.approx(0.0)
        assert "failing" in results
        assert torch.isnan(results["failing"])


class TestPointCloudMetrics:
    """Test point cloud metrics (require point-cloud-utils)."""

    @pytest.fixture
    def point_clouds(self):
        """Create sample 3D point clouds for testing."""
        preds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return preds, targets

    @pytest.fixture
    def shifted_point_clouds(self):
        """Create shifted point clouds with known distance."""
        preds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Shift all points by 0.1 in x direction
        targets = torch.tensor([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 1.0, 0.0]])
        return preds, targets

    def test_epe3d_perfect_match(self, point_clouds):
        """Test EPE3D is 0 for identical point clouds."""
        from src.metrics.point_cloud_metrics import EPE3D

        metric = EPE3D()
        preds, targets = point_clouds

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_epe3d_name(self):
        """Test EPE3D has correct name."""
        from src.metrics.point_cloud_metrics import EPE3D

        metric = EPE3D()
        assert metric.name == "epe3d"

    def test_chamfer3d_perfect_match(self, point_clouds):
        """Test Chamfer distance is 0 for identical point clouds."""
        from src.metrics.point_cloud_metrics import Chamfer3D

        metric = Chamfer3D()
        preds, targets = point_clouds

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_hausdorff_perfect_match(self, point_clouds):
        """Test Hausdorff distance is 0 for identical point clouds."""
        from src.metrics.point_cloud_metrics import HausdorffDistance

        metric = HausdorffDistance()
        preds, targets = point_clouds

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_acc3d_strict_perfect_match(self, point_clouds):
        """Test Acc3DStrict is 100% for identical point clouds."""
        from src.metrics.point_cloud_metrics import Acc3DStrict

        metric = Acc3DStrict(threshold=0.05)
        preds, targets = point_clouds

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(100.0)

    def test_acc3d_strict_shifted(self, shifted_point_clouds):
        """Test Acc3DStrict with shifted points."""
        from src.metrics.point_cloud_metrics import Acc3DStrict

        preds, targets = shifted_point_clouds

        # With threshold 0.05, all points are outside (shift is 0.1)
        metric_strict = Acc3DStrict(threshold=0.05)
        result = metric_strict.compute(preds, targets)
        assert result.item() == pytest.approx(0.0)

        # With threshold 0.2, all points are inside
        metric_relaxed = Acc3DStrict(threshold=0.2)
        result = metric_relaxed.compute(preds, targets)
        assert result.item() == pytest.approx(100.0)

    def test_max_error_3d_perfect_match(self, point_clouds):
        """Test MaxError3D is 0 for identical point clouds."""
        from src.metrics.point_cloud_metrics import MaxError3D

        metric = MaxError3D()
        preds, targets = point_clouds

        result = metric.compute(preds, targets)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_max_error_3d_shifted(self, shifted_point_clouds):
        """Test MaxError3D returns correct max distance."""
        from src.metrics.point_cloud_metrics import MaxError3D

        metric = MaxError3D()
        preds, targets = shifted_point_clouds

        result = metric.compute(preds, targets)
        # All points shifted by 0.1 in x direction
        assert result.item() == pytest.approx(0.1, abs=1e-6)
