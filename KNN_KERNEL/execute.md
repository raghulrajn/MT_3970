# KNN Kernel: Execution & Profiling Guide

This directory contains two Triton implementations of K-nearest-neighbours
over 3D point clouds — a naive O(N)-per-query brute-force kernel
(`_bruteforce_knn_kernel`) and a Morton-sorted, fixed-window approximate
kernel (`_window_knn_kernel`) — plus two scripts for evaluating them:

| Script | Purpose |
|---|---|
| [comparison.py](comparison.py) | Benchmarking & accuracy: latency, memory-locality metrics, and recall@K vs. a cuML brute-force ground truth. Sweeps tile sizes / window radius and writes results to CSV. |
| [ncu_profile.py](ncu_profile.py) | Profiling harness: runs a single kernel variant in a tight warmup+iterate loop with no Python-side timing, so `ncu` (Nsight Compute) can attach and isolate one steady-state kernel launch. |

Both load point clouds from `blank/blank_pc_{4k,16k,32k,64k}.npy`.

## Prerequisites

Run everything through `uv` from inside the sandbox container (see
[SANDBOX.md](../SANDBOX.md)) so the pinned environment (Triton, PyTorch/CUDA,
cuML, CuPy) is used:

```bash
cd /workspace          # or wherever the repo is mounted
uv sync                # first time only / after dependency changes
```

- GPU + CUDA driver available (`nvidia-smi` should work).
- `ncu` (NVIDIA Nsight Compute CLI) is only needed for the profiling workflow,
  not for `comparison.py`. Check it's on `PATH`: `ncu --version`.
- Profiling with `ncu` reads GPU performance counters, which usually requires
  elevated privileges. Inside the sandbox container you're root, so this
  should work out of the box; on a shared/cluster host you may need
  `sudo ncu ...` or an admin-enabled `NVreg_RestrictProfilingToAdminUsers=0`.

**Known gotcha:** `comparison.py`'s `SIZES` dict currently points at
absolute paths under `.../master-thesis-template-master/KNN_KERNEL/blank/...`
which only has `16k` and `1M` files on this machine, not `4k/32k/64k`. Either
regenerate/copy those `.npy` files to that path, or repoint `SIZES` at the
`./blank/` directory next to this script (as `ncu_profile.py` already does)
before running the sweep below.

## 1. Benchmarking & accuracy — `comparison.py`

Run the whole comparison from this directory:

```bash
cd KNN_KERNEL
uv run python comparison.py
```

By default the `__main__` block runs `benchmark_morton_sorted` for every
size in `SIZES`, sweeping `TILE_T` and `HALF_W` independently against a
cuML brute-force reference, and writes the combined results to
[morton_sorted_tile_halfw_sweep.csv](morton_sorted_tile_halfw_sweep.csv) in
this directory. Progress and per-setting latency/recall are printed to
stdout as it runs.

Other entry points defined in the file (not wired into `__main__` by
default — call them from a Python shell or by editing the bottom of the
file):

- `compare_one_size(size_label, pt_path)` — one-shot latency + locality +
  recall comparison between the bruteforce and Morton-windowed kernels for a
  single point-cloud size.
- `benchmark_tiling(size_label, pt_path)` — sweeps `BF_TILE_T` (bruteforce
  kernel) and `TILE_T` (windowed kernel) independently; used to produce
  [tiling_sweep_results.csv](tiling_sweep_results.csv).

Key tunables at the top of the file: `K` (neighbours), `HALF_W` (window
radius), `TILE_Q`/`TILE_T`/`BF_TILE_T` (tile sizes), `RECALL_MAX_N` (cap
above which recall is skipped since the exact cuML reference gets slow).

## 2. Profiling with Nsight Compute — `ncu_profile.py`

`ncu_profile.py` mirrors the kernels/helpers in `comparison.py` but adds a
`profile_one()` function and a CLI, meant to be driven by `ncu` rather than
run standalone for timing (its own `timed()`-based comparison still runs if
you invoke it with no `--size`/`--mode`).

### 2a. Sanity-run without ncu

```bash
cd KNN_KERNEL
uv run python ncu_profile.py --size 16k --mode sorted --warmup 3 --iters 1
```

- `--size {4k,16k,32k,64k}` — which point cloud to load.
- `--mode {sorted,unsorted}` — `sorted` = Morton-windowed kernel, `unsorted`
  = brute-force kernel.
- `--warmup` — untimed launches first (lets Triton autotune/compile and lets
  `ncu --launch-skip` skip past them).
- `--iters` — launches to run after warmup (paired with `ncu --launch-count`).

Omitting both `--size` and `--mode` instead runs `compare_one_size` for
every size in `SIZES`, same as `comparison.py`'s manual entry points, but
using the local `./blank/...` relative paths.

### 2b. Profile a single (mode, size) with ncu directly

```bash
cd KNN_KERNEL
ncu --set full \
    --kernel-name "regex:_window_knn_kernel.*" \
    --launch-skip 3 --launch-count 1 \
    -o ncu_reports/sorted_16k \
    python ncu_profile.py --size 16k --mode sorted
```

Swap the kernel regex to `regex:_bruteforce_knn_kernel.*` and
`--mode unsorted` for the brute-force kernel. `--launch-skip 3` matches the
default `warmup=3` in `profile_one`, so the profiled launch is a steady-state
one, not the first (compilation-inflated) call.

### 2c. Profile everything — `run_ncu_profile.sh`

To sweep all 4 sizes × 2 modes in one go:

```bash
cd KNN_KERNEL
./run_ncu_profile.sh
```

This writes one report per combination to `ncu_reports/`, e.g.
`ncu_reports/unsorted_4k.ncu-rep`, `ncu_reports/sorted_64k.ncu-rep`. It runs
`ncu` with `HOME=/tmp` set (works around `ncu` trying to write cache/config
files to a possibly read-only home directory) and `--set full` (collects the
full metric set — this makes each run noticeably slower than `--set basic`).

### 2d. Viewing reports

Either open the `.ncu-rep` file in the Nsight Compute GUI (copy it to a
machine with the GUI installed and `File > Open`), or dump a summary on the
CLI:

```bash
ncu --import ncu_reports/sorted_16k.ncu-rep --page details
```

## Output files in this directory

- `morton_sorted_tile_halfw_sweep.csv` — latest `TILE_T`/`HALF_W` sweep from
  `comparison.py`.
- `tiling_sweep_results.csv` — latest `BF_TILE_T`/`TILE_T` sweep from
  `benchmark_tiling`.
- `ncu_reports/*.ncu-rep` — Nsight Compute reports from `run_ncu_profile.sh`
  or manual `ncu` invocations.
