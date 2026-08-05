#!/usr/bin/env python3
"""Compile Model_ContactAware in three torch.compile modes and evaluate each
with a single-pass inference run over the DDACS point-cloud test split.

Modes: default, reduce-overhead, max-autotune.
"""

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ddacs import PointCloudDDACSDataset
from src.metrics.regression_metrics import MAE, MSE, RMSE
from src.models.model import Model_ContactAware

torch.set_float32_matmul_precision("high")

DATA_ROOT = "/mnt/ac142464/data/datasets/ddacs"
MODEL_CONFIG = ROOT_DIR / "configs/model/model.yaml"
CHECKPOINT = ROOT_DIR / "ModelArtifacts/eager.ckpt"
ARTIFACT_DIR = ROOT_DIR / "ModelArtifacts"
BATCH_SIZE = 1
NUM_SAMPLES = 10

# name -> torch.compile mode string
COMPILE_MODES = {
    "default": "default",
    "reduced_overhead": "reduce-overhead",
    "max_autotune": "max-autotune",
}


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(device: torch.device) -> Model_ContactAware:
    """Build Model_ContactAware from MODEL_CONFIG and load the eager checkpoint.

    Args:
        device: Device to move the model to before loading weights.

    Returns:
        The model in eval mode with weights restored from CHECKPOINT.
    """
    config = load_yaml(MODEL_CONFIG)
    model = Model_ContactAware(config).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model


@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device) -> dict:
    """Run a single no-grad pass over the test split and score the predictions.

    Args:
        model: Model (eager or torch.compile'd) to evaluate. Must accept the
            preprocessed input tuple returned by ``model.preprocess_data``.
        loader: DataLoader yielding raw batches from PointCloudDDACSDataset.
        device: Device to move each batch to before inference.

    Returns:
        Dict with keys "mae", "mse", "rmse" holding the metric values over
        all concatenated predictions/targets.
    """
    preds, targets = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs, target = model.preprocess_data(batch)
        pred = model(inputs)
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return {
        "mae": MAE().compute(preds, targets).item(),
        "mse": MSE().compute(preds, targets).item(),
        "rmse": RMSE().compute(preds, targets).item(),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = PointCloudDDACSDataset(DATA_ROOT, split="test")
    test_dataset = Subset(test_dataset, list(range(min(NUM_SAMPLES, len(test_dataset)))))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)} | batches: {len(test_loader)}")

    all_results = {}

    for name, mode in COMPILE_MODES.items():
        print(f"\n{'#' * 70}")
        print(f"  torch.compile mode: {mode}")
        print(f"{'#' * 70}")

        model = load_model(device)
        compiled = torch.compile(model, backend="inductor", mode=mode)

        metrics = run_inference(compiled, test_loader, device)
        all_results[name] = metrics
        print(f"MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f}")

        save_path = ARTIFACT_DIR / f"model_{name}.pt"
        torch.save(compiled, save_path)
        print(f"Saved compiled model -> {save_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Mode':<20}{'MAE':>14}{'MSE':>14}{'RMSE':>14}")
    print("-" * 70)
    for name, m in all_results.items():
        print(f"{name:<20}{m['mae']:>14.6f}{m['mse']:>14.6f}{m['rmse']:>14.6f}")
