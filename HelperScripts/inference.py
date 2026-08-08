#!/usr/bin/env python3
"""Single-pass inference across every model artifact in ModelArtifacts/.

Runs one forward pass per (model, sample) pair on user-chosen or randomly
sampled DDACS test-split simulations, using the same input layout as
compile.py / latency_profiling.py / trt_build.py. Model inputs/outputs live
in each sample's own per-simulation normalized scale; predictions and the
ground truth are denormalized back to physical mm before MAE/MSE
/RMSE are computed and before the 3-panel surface plot (ground truth |
prediction | signed error) is saved to PLOTS/.

Backends covered, all fed the SAME sample inputs:
    eager             - eager.ckpt + Model_ContactAware (see compile.py)
    eager_scripted    - model_eager.pt (TorchScript, see model_save.py)
    default / reduced_overhead / max_autotune
                      - torch.compile'd modules (see compile.py)
    onnx              - model.onnx via onnxruntime
    trt_fp32 / trt_tf32
                      - model_fp32.trt / model_tf32.trt via TensorRT (see trt_build.py)

Missing artifacts are skipped with a message rather than failing the run.

Usage
-----
    # One random test-split simulation:
    python HelperScripts/inference.py

    # 5 random simulations, reproducible:
    python HelperScripts/inference.py --num-random 5 --seed 0

    # Specific simulation IDs (from process_parameters.csv):
    python HelperScripts/inference.py --sim-ids 1023 4092 77

    # Cross-model comparison: log MAE/P50/P90/P99 per (model, sim_id) to a
    # CSV in InferenceMetrics/ instead of saving per-sim plots. Every model
    # sees the same sampled sim_ids for a fair comparison.
    python HelperScripts/inference.py --compare-models --num-random 20 --seed 0
"""

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, Subset

from utils import (
    ARTIFACT_DIR,
    DATA_ROOT,
    INPUT_NAMES,
    OUTPUT_NAMES,
    compute_metrics,
    get_device,
    load_compiled_model,
    load_eager_model,
    load_scripted_model,
    prepare_inputs,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ddacs import ExportDDACSDataset

PLOTS_DIR = ROOT_DIR / "PLOTS"
INFERENCE_METRICS_DIR = ROOT_DIR / "InferenceMetrics"

# model key -> (artifact path, loader(path, device))
MODEL_ARTIFACTS = {
    "eager":            (ARTIFACT_DIR / "eager.ckpt",                lambda p, d: load_eager_model(d)),
    "eager_scripted":   (ARTIFACT_DIR / "model_eager.pt",            lambda p, d: load_scripted_model(p, d)),
    "default":          (ARTIFACT_DIR / "model_default.pt",          lambda p, d: load_compiled_model(p, d)),
    "reduced_overhead": (ARTIFACT_DIR / "model_reduced_overhead.pt", lambda p, d: load_compiled_model(p, d)),
    "max_autotune":     (ARTIFACT_DIR / "model_max_autotune.pt",     lambda p, d: load_compiled_model(p, d)),
    "onnx":             (ARTIFACT_DIR / "model.onnx",                lambda p, d: load_onnx_session(p, d)),
    "trt_fp32":         (ARTIFACT_DIR / "model_fp32.trt",            lambda p, d: TRTRunner(p, d)),
    "trt_tf32":         (ARTIFACT_DIR / "model_tf32.trt",            lambda p, d: TRTRunner(p, d)),
}


class TRTRunner:
    """Runs single-pass inference through a serialized TensorRT engine.

    The engine is deserialized once at construction. Each call to
    :meth:`infer` is synchronous on its own CUDA stream and returns a
    ``(1, N, 3)`` float32 CUDA tensor.
    """

    def __init__(self, engine_path: Path, device: torch.device) -> None:
        """Deserialize the engine and catalogue its I/O tensor names.

        Args:
            engine_path: Path to the serialized ``.trt`` engine file.
            device: CUDA device to run the engine on.
        """
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        self.device = device
        self._engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device)

        self._input_names: List[str] = []
        self._output_names: List[str] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            (self._input_names if mode == trt.TensorIOMode.INPUT else self._output_names).append(name)

    @torch.no_grad()
    def infer(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run one forward pass through the engine.

        Args:
            *inputs: 8 tensors in INPUT_NAMES order, as returned by
                ``prepare_inputs``.

        Returns:
            Predicted blank3 tensor, shape (1, N, 3), float32, on ``device``.

        Raises:
            RuntimeError: If engine execution fails.
        """
        named = dict(zip(INPUT_NAMES, inputs))
        live: Dict[str, torch.Tensor] = {}
        for name in self._input_names:
            t = named[name].to(device=self.device, dtype=torch.float32).contiguous()
            live[name] = t
            self._context.set_input_shape(name, tuple(t.shape))
            self._context.set_tensor_address(name, t.data_ptr())

        out_name = self._output_names[0]
        out_shape = tuple(self._context.get_tensor_shape(out_name))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self._context.set_tensor_address(out_name, out.data_ptr())

        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False — engine execution failed")
        self._stream.synchronize()
        return out


def load_onnx_session(path: Path, device: torch.device) -> ort.InferenceSession:
    """Open an ONNX Runtime inference session for a model.onnx graph.

    Args:
        path: Path to the ONNX graph (see trt_build.py's export_onnx).
        device: If CUDA, CUDAExecutionProvider is tried first (falls back
            to CPU automatically if unavailable).

    Returns:
        The inference session.
    """
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def prepare_target(batch: dict, device: torch.device) -> torch.Tensor:
    """Extract the ground-truth final blank state, matching preprocess_data's target.

    Args:
        batch: Raw batch dict with a "blank" tensor of shape (B, T>=4, N, 3).
        device: Device to move the tensor to.

    Returns:
        Ground-truth blank3 tensor, shape (B, N, 3), float, on device.
    """
    return batch["blank"][:, 3].to(device).contiguous()


def prepare_denorm(batch: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the per-sample denormalization scale/offset.

    Every point cloud in a sample (blank/die/punch/binder, all timesteps) is
    normalized into that sample's own canonical space; ``norm_scale`` /
    ``norm_offset`` invert it back to physical mm via
    ``real = normalized * scale + offset``.

    Args:
        batch: Raw batch dict with "norm_scale" (B, 1) and "norm_offset"
            (B, 3) tensors, as yielded by the DataLoader.
        device: Device to move each tensor to.

    Returns:
        Tuple ``(scale, offset)`` of shape ``(B, 1)`` and ``(B, 3)``, on device.
    """
    return batch["norm_scale"].to(device), batch["norm_offset"].to(device)


def denormalize(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Invert per-sample point-cloud normalization back to physical mm.

    Args:
        x: Normalized point cloud, shape (B, N, 3).
        scale: Per-sample scale, shape (B, 1), as returned by ``prepare_denorm``.
        offset: Per-sample offset, shape (B, 3), as returned by ``prepare_denorm``.

    Returns:
        Denormalized point cloud in mm, shape (B, N, 3).
    """
    return x * scale.view(-1, 1, 1) + offset.view(-1, 1, 3)


def run_inference(model, inputs: tuple) -> torch.Tensor:
    """Run one forward pass through any ModelArtifacts backend.

    Args:
        model: Loaded eager/scripted/compiled nn.Module, onnxruntime
            InferenceSession, or TRTRunner.
        inputs: 8-tuple as returned by ``prepare_inputs``.

    Returns:
        Predicted blank3 tensor, shape (1, N, 3), float.
    """
    if isinstance(model, ort.InferenceSession):
        feed = {name: t.detach().cpu().numpy() for name, t in zip(INPUT_NAMES, inputs)}
        out = model.run(OUTPUT_NAMES, feed)[0]
        return torch.from_numpy(out)
    if isinstance(model, TRTRunner):
        return model.infer(*inputs)
    with torch.no_grad():
        return model(inputs)


def select_sample_indices(
    dataset: ExportDDACSDataset,
    sim_ids: List[int] | None,
    num_random: int,
    seed: int | None,
) -> List[int]:
    """Resolve which dataset indices to run inference on.

    Args:
        dataset: Test-split point-cloud ExportDDACSDataset.
        sim_ids: Explicit simulation IDs to run, or None for random sampling.
        num_random: Number of random samples to draw when sim_ids is None.
        seed: Random seed for the random draw (None = nondeterministic).

    Returns:
        List of dataset indices, in the order the results should be reported.

    Raises:
        ValueError: If any requested sim_id isn't in the test split.
    """
    if sim_ids:
        idx_by_sim_id = {int(s): i for i, s in enumerate(dataset._selected_sim_ids)}
        missing = [s for s in sim_ids if s not in idx_by_sim_id]
        if missing:
            raise ValueError(f"sim_ids not found in the test split: {missing}")
        return [idx_by_sim_id[s] for s in sim_ids]
    rng = random.Random(seed)
    return rng.sample(range(len(dataset)), k=min(num_random, len(dataset)))


def compute_error_percentiles(pred_mm: np.ndarray, gt_mm: np.ndarray) -> Dict[str, float]:
    """Per-point Euclidean error percentiles (mm), matching _plot_springback's MAE.

    Args:
        pred_mm: Predicted point cloud in mm, shape (N, 3).
        gt_mm:   Ground-truth point cloud in mm, shape (N, 3).

    Returns:
        Dict with keys "p50", "p90", "p99".
    """
    err = np.linalg.norm(pred_mm - gt_mm, axis=1)
    return {
        "p50": float(np.percentile(err, 50)),
        "p90": float(np.percentile(err, 90)),
        "p99": float(np.percentile(err, 99)),
    }


_AXIS_LIMITS: Tuple[float, float] = (-1.0, 1.0)


def _set_axis_limits(gt_mm: np.ndarray, pad_frac: float = 0.05) -> None:
    """Recompute the shared cube axis limits used by _plot_springback.

    Call once per sample (from the ground truth alone) so every model's
    plot for that sample shares identical framing/scale.

    Args:
        gt_mm: Ground-truth point cloud in mm, shape (N, 3).
        pad_frac: Fractional padding added around the data's min/max extent.
    """
    global _AXIS_LIMITS
    lo, hi = float(gt_mm.min()), float(gt_mm.max())
    pad = pad_frac * (hi - lo)
    _AXIS_LIMITS = (lo - pad, hi + pad)


def _plot_springback(
    pred_mm: np.ndarray,
    gt_mm: np.ndarray,
    sim_id: int,
    model_key: str,
    save_path: Path,
) -> None:
    """3-panel figure: GT | Pred | Signed Error heatmap plotted as surfaces.

    Args:
        pred_mm:    Predicted point cloud in mm, shape (4096, 3).
        gt_mm:      Ground-truth point cloud in mm, shape (4096, 3).
        sim_id:     Simulation ID for the figure title.
        model_key:  Model key for the figure title.
        save_path:  Output file path.
    """
    unsigned_err = np.linalg.norm(pred_mm - gt_mm, axis=1)
    mae = float(unsigned_err.mean())

    grid_res = 100

    x_min, x_max = gt_mm[:, 0].min(), gt_mm[:, 0].max()
    y_min, y_max = gt_mm[:, 1].min(), gt_mm[:, 1].max()

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )

    grid_z_gt = griddata(gt_mm[:, :2], gt_mm[:, 2], (grid_x, grid_y), method='cubic')
    grid_z_pred = griddata(pred_mm[:, :2], pred_mm[:, 2], (grid_x, grid_y), method='cubic')

    # Calculate SIGNED error on the Z grid (Pred - GT)
    # Negative = Pred is below GT (Blue)
    # Zero     = Pred matches GT (Green)
    # Positive = Pred is above GT (Red)
    grid_err = grid_z_pred - grid_z_gt
    max_abs_err = np.nanmax(np.abs(grid_err))
    vmax = float(max_abs_err) if max_abs_err > 0 else 1.0
    vmax=1
    vmin = -vmax

    grid_err_clipped = np.clip(grid_err, vmin, vmax)

    bgr_cmap = LinearSegmentedColormap.from_list("BlueGreenRed", ["blue", "green", "red"])
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"DDACS | { sim_id } | MAE: {mae:.4f} mm",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    # Panel 1: Ground Truth Surface
    ax1.plot_surface(grid_x, grid_y, grid_z_gt, alpha=0.8, edgecolor='none')
    ax1.set_title("Ground Truth Surface")

    # Panel 2: Prediction Surface
    ax2.plot_surface(grid_x, grid_y, grid_z_pred, alpha=0.8, edgecolor='none')
    ax2.set_title("Prediction Surface")

    # Panel 3: Error Heatmap Surface
    rcount, ccount = grid_err_clipped.shape

    sc = ax3.plot_surface(
        grid_x, grid_y, grid_z_gt,
        facecolors=bgr_cmap(norm(grid_err_clipped)),
        alpha=0.9,
        edgecolor='none',
        rcount=rcount,
        ccount=ccount
    )

    sm = plt.cm.ScalarMappable(cmap=bgr_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, shrink=0.6)
    cbar.set_label("Signed Z-Displacement Error (mm)")
    ax3.set_title("Error Heatmap Surface")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(_AXIS_LIMITS)
        ax.set_ylim(_AXIS_LIMITS)
        ax.set_zlim(_AXIS_LIMITS)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-pass inference across every ModelArtifacts/ backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sim-ids", type=int, nargs="+", default=None,
        help="Specific test-split simulation IDs to run. Default: random file(s).",
    )
    p.add_argument(
        "--num-random", type=int, default=1,
        help="Number of random test-split files to sample when --sim-ids is not given.",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for --num-random sampling.")
    p.add_argument(
        "--compare-models", action="store_true",
        help="Log per-(model, sim_id) MAE/P50/P90/P99 to a CSV in InferenceMetrics/ for "
             "cross-model comparison, using the same sampled sim_ids for every model. "
             "Skips per-sim plots.",
    )
    p.add_argument(
        "--csv-name", type=str, default="inference_comparison.csv",
        help="CSV filename written to InferenceMetrics/ when --compare-models is set.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_writer = None
    csv_file = None
    csv_path = None
    if args.compare_models:
        INFERENCE_METRICS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = INFERENCE_METRICS_DIR / args.csv_name
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["Model", "SIM_ID", "MAE", "MSE", "RMSE", "P50", "P90", "P99"]
        )
        csv_writer.writeheader()
        print(f"Logging comparison metrics to {csv_path}")

    test_dataset = ExportDDACSDataset(
        DATA_ROOT, "pointcloud", split="test",
        fields=["blank", "die", "punch", "binder", "norm_scale", "norm_offset"],
    )
    sample_indices = select_sample_indices(test_dataset, args.sim_ids, args.num_random, args.seed)
    sim_ids = [int(test_dataset._selected_sim_ids[i]) for i in sample_indices]
    print(f"Running inference on sim_ids: {sim_ids}")

    subset_loader = DataLoader(Subset(test_dataset, sample_indices), batch_size=1, shuffle=False)
    batches = list(subset_loader)

    all_results: Dict[Tuple[str, int], dict] = {}

    for model_key, (path, loader_fn) in MODEL_ARTIFACTS.items():
        if not path.exists():
            print(f"\nSkipping '{model_key}': artifact not found at {path}")
            continue

        print(f"\n{'#' * 70}\n  Model: {model_key}\n{'#' * 70}")
        try:
            model = loader_fn(path, device)
        except Exception as exc:
            print(f"Skipping '{model_key}': failed to load ({exc!r})")
            continue

        for sim_id, batch in zip(sim_ids, batches):
            inputs = prepare_inputs(batch, device)
            target = prepare_target(batch, device)
            scale, offset = prepare_denorm(batch, device)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = run_inference(model, inputs).to(device)

            pred_denorm = denormalize(pred, scale, offset)
            target_denorm = denormalize(target, scale, offset)

            metrics = compute_metrics(pred_denorm, target_denorm)
            all_results[(model_key, sim_id)] = metrics
            print(f"  sim_id={sim_id}: MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f} (mm)")

            pred_mm = pred_denorm.squeeze(0).detach().cpu().numpy()
            gt_mm = target_denorm.squeeze(0).detach().cpu().numpy()

            if csv_writer is not None:
                pct = compute_error_percentiles(pred_mm, gt_mm)
                csv_writer.writerow({
                    "Model": model_key,
                    "SIM_ID": sim_id,
                    "MAE": metrics["mae"],
                    "MSE": metrics["mse"],
                    "RMSE": metrics["rmse"],
                    "P50": pct["p50"],
                    "P90": pct["p90"],
                    "P99": pct["p99"],
                })
                csv_file.flush()
            else:
                _set_axis_limits(gt_mm)
                save_path = PLOTS_DIR / f"{sim_id}_{model_key}.png"
                _plot_springback(pred_mm, gt_mm, sim_id, model_key, save_path)
                print(f"  Saved plot -> {save_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if csv_file is not None:
        csv_file.close()
        print(f"Comparison metrics saved to {csv_path}")

    # ---- Summary table -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20}{'sim_id':>10}{'MAE':>14}{'MSE':>14}{'RMSE':>14}")
    print("-" * 70)
    for (model_key, sim_id), m in all_results.items():
        print(f"{model_key:<20}{sim_id:>10}{m['mae']:>14.6f}{m['mse']:>14.6f}{m['rmse']:>14.6f}")
