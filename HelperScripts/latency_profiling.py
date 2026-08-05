#!/usr/bin/env python3
"""Latency profiling for every model artifact in ModelArtifacts/ on the DDACS
point-cloud test split.

For each artifact (eager checkpoint, TorchScript-eager, and the three
torch.compile modes saved by compile.py): 100 samples x 10 timed inferences
(1000 total) after a proper untimed warm-up, reporting mean / p50 / p90
latency. Also captures a torch.profiler execution trace for one inference
pass per model, saved to its own subfolder under ModelArtifacts/traces/.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.profiler
from torch.utils.data import DataLoader, Subset

from utils import (
    ARTIFACT_DIR,
    DATA_ROOT,
    get_device,
    load_compiled_model,
    load_eager_model,
    load_scripted_model,
    prepare_inputs,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.ddacs import PointCloudDDACSDataset

TRACE_DIR = ARTIFACT_DIR / "traces"

NUM_FILES = 10
RUNS_PER_FILE = 10
WARMUP_RUNS = 20

# label -> (artifact path, loader taking (path, device))
MODEL_ARTIFACTS = {
    "eager":            (ARTIFACT_DIR / "eager.ckpt",                lambda p, d: load_eager_model(d)),
    "eager_scripted":   (ARTIFACT_DIR / "model_eager.pt",            load_scripted_model),
    "default":          (ARTIFACT_DIR / "model_default.pt",          load_compiled_model),
    "reduced_overhead": (ARTIFACT_DIR / "model_reduced_overhead.pt", load_compiled_model),
    "max_autotune":     (ARTIFACT_DIR / "model_max_autotune.pt",     load_compiled_model),
}


def print_latency_stats(name: str, latencies_ms: list) -> dict:
    """Compute mean/p50/p90 latency and print a formatted summary block.

    Args:
        name: Label of the model artifact being reported (e.g. "eager").
        latencies_ms: Per-inference latencies in milliseconds.

    Returns:
        Dict with keys "mean", "p50", "p90" holding the computed statistics.
    """
    lat = np.array(latencies_ms)
    stats = {"mean": lat.mean(), "p50": np.percentile(lat, 50), "p90": np.percentile(lat, 90)}
    print(f"\n{'=' * 70}")
    print(f"LATENCY STATISTICS (ms) - {name} - {len(lat)} inferences "
          f"({NUM_FILES} files x {RUNS_PER_FILE} runs)")
    print(f"{'=' * 70}")
    print(f"  Mean : {stats['mean']:.4f}")
    print(f"  P50  : {stats['p50']:.4f}")
    print(f"  P90  : {stats['p90']:.4f}")
    print(f"{'=' * 70}\n")
    return stats


def profile_model(name: str, model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Warm up, timed-profile, and trace a single model artifact.

    Runs WARMUP_RUNS untimed forward passes, then times RUNS_PER_FILE
    inferences per file over the loader (NUM_FILES x RUNS_PER_FILE total)
    using CUDA events, and finally captures a torch.profiler execution trace
    for one inference pass, saved under ``TRACE_DIR / name``.

    Args:
        name: Label of the model artifact (used for printing and the trace
            output subfolder).
        model: Loaded model (eager, TorchScript, or compiled) to profile.
        loader: DataLoader yielding raw batches from PointCloudDDACSDataset.
        device: CUDA device the model and inputs live on.

    Returns:
        Dict with keys "mean", "p50", "p90" holding latency statistics in
        milliseconds, as returned by ``print_latency_stats``.
    """
    print(f"\n{'#' * 70}\n  Profiling: {name}\n{'#' * 70}")

    # ---- Proper warm-up (untimed) ------------------------------------------
    warmup_inputs = prepare_inputs(next(iter(loader)), device)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(warmup_inputs)
        torch.cuda.synchronize()
    print(f"Warm-up complete ({WARMUP_RUNS} untimed passes)")

    # ---- Timed loop: NUM_FILES x RUNS_PER_FILE inferences ------------------
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS_PER_FILE)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS_PER_FILE)]

    latencies_ms = []
    with torch.no_grad():
        for file_idx, batch in enumerate(loader):
            inputs = prepare_inputs(batch, device)

            for r in range(RUNS_PER_FILE):
                starts[r].record()
                _ = model(inputs)
                ends[r].record()
            torch.cuda.synchronize()

            file_latencies = [starts[r].elapsed_time(ends[r]) for r in range(RUNS_PER_FILE)]
            latencies_ms.extend(file_latencies)
            print(f"  File {file_idx + 1}/{len(loader)}: mean {np.mean(file_latencies):.4f} ms")

    stats = print_latency_stats(name, latencies_ms)

    # ---- Execution trace for a single inference pass ------------------------
    trace_dir = TRACE_DIR / name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_inputs = prepare_inputs(next(iter(loader)), device)
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            _ = model(trace_inputs)
            torch.cuda.synchronize()

    trace_path = trace_dir / "single_inference_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"Saved execution trace -> {trace_path}")

    return stats


if __name__ == "__main__":
    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    dataset = PointCloudDDACSDataset(DATA_ROOT, split="test")
    dataset = Subset(dataset, list(range(min(NUM_FILES, len(dataset)))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(dataset)} | batches: {len(loader)}")

    all_stats = {}
    for name, (path, loader_fn) in MODEL_ARTIFACTS.items():
        if not path.exists():
            print(f"\nSkipping '{name}': artifact not found at {path}")
            continue
        try:
            model = loader_fn(path, device)
            all_stats[name] = profile_model(name, model, loader, device)
        except Exception as exc:
            print(f"\nSkipping '{name}': failed to load/profile ({exc!r})")
        finally:
            del model
            torch.cuda.empty_cache()

    # ---- Summary table -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<20}{'Mean (ms)':>14}{'P50 (ms)':>14}{'P90 (ms)':>14}")
    print("-" * 70)
    for name, s in all_stats.items():
        print(f"{name:<20}{s['mean']:>14.4f}{s['p50']:>14.4f}{s['p90']:>14.4f}")
