#!/usr/bin/env python3
"""Shared helpers for the ModelArtifacts pipeline scripts.

Common path conventions, model loaders, input prep, and metric scoring used
by compile.py, latency_profiling.py, trt_build.py, and inference.py -- kept
in one place so the four stay consistent with each other instead of drifting.
"""

import sys
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.metrics.regression_metrics import MAE, MSE, RMSE
from src.models.model import Model_ContactAware

torch.set_float32_matmul_precision("high")

DATA_ROOT = "/mnt/ac142464/data/datasets/ddacs"
MODEL_CONFIG = ROOT_DIR / "configs/model/model.yaml"
ARTIFACT_DIR = ROOT_DIR / "ModelArtifacts"
CHECKPOINT = ARTIFACT_DIR / "eager.ckpt"

# Flat 8-tensor model input layout shared by every ModelArtifacts backend
# (eager, scripted, compiled, ONNX, TensorRT) -- see prepare_inputs.
INPUT_NAMES = ["blank0", "die1", "punch1", "binder1", "die2", "punch2", "binder2", "params"]
OUTPUT_NAMES = ["blank3_hat"]


def get_device(require_cuda: bool = False) -> torch.device:
    """Pick the compute device.

    Args:
        require_cuda: If True, raise instead of silently falling back to CPU.

    Returns:
        A CUDA device if available, otherwise CPU (unless require_cuda).

    Raises:
        RuntimeError: If require_cuda is True and no CUDA device is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA required but not available")
    return device


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eager_model(
    device: torch.device,
    config_path: Path = MODEL_CONFIG,
    checkpoint_path: Path = CHECKPOINT,
) -> Model_ContactAware:
    """Build Model_ContactAware from a config and load an eager checkpoint.

    Args:
        device: Device to move the model to before loading weights.
        config_path: Path to the model's YAML config.
        checkpoint_path: Path to the eager checkpoint (.ckpt).

    Returns:
        The model in eval mode with weights restored from checkpoint_path.
    """
    config = load_yaml(config_path)
    model = Model_ContactAware(config).to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model


def load_scripted_model(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load a TorchScript module from disk.

    Args:
        path: Path to a ``torch.jit.save``d module.
        device: Device to map the module's tensors onto.

    Returns:
        The scripted module in eval mode.
    """
    return torch.jit.load(str(path), map_location=device).eval()


def load_compiled_model(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a torch.compile'd model saved via ``torch.save`` (compile.py).

    Args:
        path: Path to the pickled model object.
        device: Device to map the model's tensors onto.

    Returns:
        The compiled model in eval mode.
    """
    model = torch.load(path, map_location=device, weights_only=False)
    return model.eval()


def prepare_inputs(batch: dict, device: torch.device) -> tuple:
    """Build the 8-tensor model input tuple from a raw dataset batch.

    Mirrors Model_ContactAware.preprocess_data: extracts blank0 and the
    die/punch/binder tool clouds at timesteps T1/T2 plus process parameters,
    in the layout every ModelArtifacts backend (eager, scripted, compiled,
    ONNX, TensorRT) expects, in INPUT_NAMES order.

    Args:
        batch: Raw batch dict with "blank", "die", "punch", "binder", and
            "parameters" tensors, as yielded by the DataLoader.
        device: Device to move each tensor to.

    Returns:
        Tuple of contiguous tensors ``(blank0, die1, punch1, binder1, die2,
        punch2, binder2, parameters)``.
    """
    blank, die, punch, binder = batch["blank"], batch["die"], batch["punch"], batch["binder"]
    inputs = (
        blank[:, 0].to(device).contiguous(),
        die[:, 1].to(device).contiguous(), punch[:, 1].to(device).contiguous(), binder[:, 1].to(device).contiguous(),
        die[:, 2].to(device).contiguous(), punch[:, 2].to(device).contiguous(), binder[:, 2].to(device).contiguous(),
        batch["parameters"].to(device).contiguous(),
    )
    return inputs


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Score a prediction against its ground truth.

    Args:
        pred: Predicted tensor.
        target: Ground-truth tensor, same shape as pred.

    Returns:
        Dict with keys "mae", "mse", "rmse".
    """
    pred_cpu = pred.detach().cpu().float()
    target_cpu = target.detach().cpu().float()
    return {
        "mae": MAE().compute(pred_cpu, target_cpu).item(),
        "mse": MSE().compute(pred_cpu, target_cpu).item(),
        "rmse": RMSE().compute(pred_cpu, target_cpu).item(),
    }
