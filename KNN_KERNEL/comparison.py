import os
import re
import csv
import time
import statistics
import numpy as np
import cupy as cp
import torch
import triton
import triton.language as tl
from cuml.neighbors import NearestNeighbors as cuMLNearestNeighbors

@triton.jit
def _bruteforce_knn_kernel(
    Pts_ptr,
    OutIdx_ptr,
    N,
    K:      tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_T: tl.constexpr,
    stride_pts_b, stride_pts_n, stride_pts_d,
    stride_out_b, stride_out_n, stride_out_k,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_offset = pid_q * TILE_Q
    offs_ql  = tl.arange(0, TILE_Q)
    offs_qa  = q_offset + offs_ql
    mask_q   = offs_qa < N

    base_q = Pts_ptr + pid_b * stride_pts_b + offs_qa * stride_pts_n
    q0 = tl.load(base_q + 0 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)
    q1 = tl.load(base_q + 1 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)
    q2 = tl.load(base_q + 2 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)

    topk_val = tl.full((TILE_Q, K), float('inf'), dtype=tl.float32)
    topk_idx = tl.zeros((TILE_Q, K), dtype=tl.int32)
    k_range  = tl.arange(0, K)

    # Loop over the ENTIRE point array in chunks of TILE_T.
    # N is a runtime value here (not tl.constexpr), so this compiles to a
    # genuine GPU-side loop, not an unrolled compile-time expansion -
    # same pattern as the standard Triton matmul tutorial's K-loop.
    for t_blk in range(0, N, TILE_T):
        offs_tl = t_blk + tl.arange(0, TILE_T)
        mask_t  = offs_tl < N

        base_t = Pts_ptr + pid_b * stride_pts_b + offs_tl * stride_pts_n
        t0 = tl.load(base_t + 0 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)
        t1 = tl.load(base_t + 1 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)
        t2 = tl.load(base_t + 2 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)

        d0 = q0[:, None] - t0[None, :]
        d1 = q1[:, None] - t1[None, :]
        d2 = q2[:, None] - t2[None, :]
        dist = d0*d0 + d1*d1 + d2*d2

        is_self = offs_tl[None, :] == offs_qa[:, None]
        invalid = ~(mask_q[:, None] & mask_t[None, :]) | is_self
        dist    = tl.where(invalid, float('inf'), dist)

        cand_idx = tl.cast(offs_tl, tl.int32)
        t_range  = tl.arange(0, TILE_T)

        for _ in range(K):
            min_d = tl.min(dist, axis=1)

            is_min    = dist == min_d[:, None]
            first_pos = tl.min(
                tl.where(is_min, t_range[None, :], TILE_T), axis=1
            )
            is_first  = t_range[None, :] == first_pos[:, None]
            min_i     = tl.sum(
                tl.where(is_first, cand_idx[None, :], 0), axis=1
            )

            cur_max     = tl.max(topk_val, axis=1)
            should_ins  = min_d < cur_max
            is_max_slot = topk_val == cur_max[:, None]
            evict_pos   = tl.min(
                tl.where(is_max_slot, k_range[None, :], K), axis=1
            )
            replace  = (k_range[None, :] == evict_pos[:, None]) & should_ins[:, None]

            topk_val = tl.where(replace, min_d[:, None], topk_val)
            topk_idx = tl.where(replace, min_i[:, None], topk_idx)

            dist = tl.where(is_first, float('inf'), dist)

    tl.store(
        OutIdx_ptr
        + pid_b * stride_out_b
        + offs_qa[:, None] * stride_out_n
        + k_range[None, :] * stride_out_k,
        topk_idx,
        mask=mask_q[:, None],
    )


def bruteforce_knn_triton(points: torch.Tensor, k: int, tile_q: int = 16, tile_t: int = 256) -> torch.Tensor:
    B, N, D = points.shape
    assert D == 3, f"expected D=3, got D={D}"

    out  = torch.empty(B, N, k, dtype=torch.int32, device=points.device)
    grid = (triton.cdiv(N, tile_q), B)

    _bruteforce_knn_kernel[grid](
        points, out, N, k, tile_q, tile_t,
        points.stride(0), points.stride(1), points.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        num_warps=8, num_stages=2,
    )
    return out

@triton.jit
def _window_knn_kernel(
    Pts_ptr,
    OutIdx_ptr,
    N,
    K:      tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_T: tl.constexpr,
    HALF_W: tl.constexpr,
    stride_pts_b, stride_pts_n, stride_pts_d,
    stride_out_b, stride_out_n, stride_out_k,
):
    T_WIN: tl.constexpr = TILE_Q + 2 * HALF_W

    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_offset = pid_q * TILE_Q
    t_start  = q_offset - HALF_W

    offs_ql = tl.arange(0, TILE_Q)
    offs_qa = q_offset + offs_ql
    mask_q  = offs_qa < N

    base_q = Pts_ptr + pid_b * stride_pts_b + offs_qa * stride_pts_n
    q0 = tl.load(base_q + 0 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)
    q1 = tl.load(base_q + 1 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)
    q2 = tl.load(base_q + 2 * stride_pts_d, mask=mask_q, other=0.0).to(tl.float32)

    topk_val = tl.full((TILE_Q, K), float('inf'), dtype=tl.float32)
    topk_idx = tl.zeros((TILE_Q, K), dtype=tl.int32)
    k_range  = tl.arange(0, K)

    for t_blk in range(0, T_WIN, TILE_T):
        offs_tl = t_blk + tl.arange(0, TILE_T)
        offs_ta = t_start + offs_tl

        offs_ta_safe = tl.minimum(tl.maximum(offs_ta, 0), N - 1)
        mask_t = (offs_ta >= 0) & (offs_ta < N) & (offs_tl < T_WIN)

        base_t = Pts_ptr + pid_b * stride_pts_b + offs_ta_safe * stride_pts_n
        t0 = tl.load(base_t + 0 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)
        t1 = tl.load(base_t + 1 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)
        t2 = tl.load(base_t + 2 * stride_pts_d, mask=mask_t, other=0.0).to(tl.float32)

        d0 = q0[:, None] - t0[None, :]
        d1 = q1[:, None] - t1[None, :]
        d2 = q2[:, None] - t2[None, :]
        dist = d0*d0 + d1*d1 + d2*d2

        in_window = (offs_ta[None, :] >= offs_qa[:, None] - HALF_W) & (offs_ta[None, :] <= offs_qa[:, None] + HALF_W)
        is_self   = offs_tl[None, :] == (offs_ql[:, None] + HALF_W)
        invalid   = ~(mask_q[:, None] & mask_t[None, :] & in_window) | is_self
        dist      = tl.where(invalid, float('inf'), dist)

        cand_idx = tl.cast(offs_ta_safe, tl.int32)
        t_range  = tl.arange(0, TILE_T)

        for _ in range(K):
            min_d = tl.min(dist, axis=1)

            is_min    = dist == min_d[:, None]
            first_pos = tl.min(
                tl.where(is_min, t_range[None, :], TILE_T), axis=1
            )
            is_first  = t_range[None, :] == first_pos[:, None]
            min_i     = tl.sum(
                tl.where(is_first, cand_idx[None, :], 0), axis=1
            )

            cur_max     = tl.max(topk_val, axis=1)
            should_ins  = min_d < cur_max
            is_max_slot = topk_val == cur_max[:, None]
            evict_pos   = tl.min(
                tl.where(is_max_slot, k_range[None, :], K), axis=1
            )
            replace  = (k_range[None, :] == evict_pos[:, None]) & should_ins[:, None]

            topk_val = tl.where(replace, min_d[:, None], topk_val)
            topk_idx = tl.where(replace, min_i[:, None], topk_idx)

            dist = tl.where(is_first, float('inf'), dist)

    tl.store(
        OutIdx_ptr
        + pid_b * stride_out_b
        + offs_qa[:, None] * stride_out_n
        + k_range[None, :] * stride_out_k,
        topk_idx,
        mask=mask_q[:, None],
    )


def morton_window_knn_triton(
    points: torch.Tensor,
    k: int,
    half_w: int = 256,
    tile_q: int = 16,
    tile_t: int = 32,
    morton_bits: int = 16,
) -> torch.Tensor:
    B, N, D = points.shape
    assert D == 3, f"expected D=3, got D={D}"
    assert k <= 2 * half_w + 1

    sorted_points, perm, inv_perm = morton_sort(points, bits_per_dim=morton_bits)

    out  = torch.empty(B, N, k, dtype=torch.int32, device=points.device)
    grid = (triton.cdiv(N, tile_q), B)

    _window_knn_kernel[grid](
        sorted_points, out, N, k, tile_q, tile_t, half_w,
        sorted_points.stride(0), sorted_points.stride(1), sorted_points.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        num_warps=8, num_stages=2,
    )

    B_, N_, K_ = out.shape
    out_orig  = torch.gather(perm.unsqueeze(-1).expand(-1, -1, K_), 1, out.to(torch.int64))
    out_final = torch.gather(out_orig, 1, inv_perm.unsqueeze(-1).expand(-1, -1, K_))
    return out_final.to(torch.int32)


# ── morton_sort helper (unchanged from your original code) ─────────────────

def _spread_bits_3(x: torch.Tensor, bits: int) -> torch.Tensor:
    x = x.to(torch.int64) & ((1 << bits) - 1)
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def morton_encode_3d(points: torch.Tensor, bits_per_dim: int = 16) -> torch.Tensor:
    mins = points.min(dim=0, keepdim=True).values
    maxs = points.max(dim=0, keepdim=True).values
    extent = (maxs - mins).clamp_min(1e-12)
    scale = (1 << bits_per_dim) - 1
    q = ((points - mins) / extent * scale).round().clamp(0, scale).to(torch.int64)
    xs = _spread_bits_3(q[:, 0], bits_per_dim)
    ys = _spread_bits_3(q[:, 1], bits_per_dim)
    zs = _spread_bits_3(q[:, 2], bits_per_dim)
    return (xs << 2) | (ys << 1) | zs


def morton_sort(points: torch.Tensor, bits_per_dim: int = 16):
    B, N, _ = points.shape
    perm = torch.empty(B, N, dtype=torch.int64, device=points.device)
    for b in range(B):
        codes = morton_encode_3d(points[b], bits_per_dim=bits_per_dim)
        perm[b] = torch.argsort(codes)
    sorted_points = torch.gather(points, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
    inv_perm = torch.argsort(perm, dim=1)
    return sorted_points, perm, inv_perm


SIZES = {
    "4k":  "/home/RUS_CIP/st189432/master-thesis-template-master/KNN_KERNEL/blank/blank_pc_4K.npy",
    "16k": "/home/RUS_CIP/st189432/master-thesis-template-master/KNN_KERNEL/blank/blank_pc_16K.npy",
    "32k": "/home/RUS_CIP/st189432/master-thesis-template-master/KNN_KERNEL/blank/blank_pc_32K.npy",
    "64k": "/home/RUS_CIP/st189432/master-thesis-template-master/KNN_KERNEL/blank/blank_pc_64K.npy",
}

K       = 16
HALF_W  = 1024
TILE_Q  = 16
TILE_T  = 32       # windowed kernel tile
BF_TILE_T = 1024    # bruteforce kernel tile (bigger, since it scans all N)

# recall is only affordable up to this size with the O(N) exhaustive kernel
# used as ground truth; above this, recall is skipped and only timing +
# locality metrics are reported. Raise if you have the patience/time budget.
RECALL_MAX_N = 200_000


def recall_at_k(approx: torch.Tensor, exact: torch.Tensor) -> float:
    matches = (approx.sort(-1).values.unsqueeze(-1)
               == exact.sort(-1).values.unsqueeze(-2)).any(-2)
    return matches.float().mean().item()


def cuml_bruteforce_knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Ground-truth exact KNN via RAPIDS cuML brute force.

    Independent of `_bruteforce_knn_kernel` above, so it can be used as the
    recall reference for BOTH the unsorted and sorted kernels without the
    circularity of comparing a kernel against itself.
    """
    B, N, _ = points.shape
    out = torch.empty(B, N, k, dtype=torch.int32, device=points.device)
    row_idx = torch.arange(N, device=points.device).unsqueeze(1)
    col_idx = torch.arange(k + 1, device=points.device).unsqueeze(0)

    for b in range(B):
        nn = cuMLNearestNeighbors(
            n_neighbors=k + 1, algorithm="brute", metric="euclidean",
            output_type="cupy",
        )
        pts_cp = cp.from_dlpack(points[b].contiguous())
        nn.fit(pts_cp)
        _, idx = nn.kneighbors(pts_cp)   # (N, k+1), includes self at dist 0
        idx = torch.from_dlpack(idx)

        # drop the self-match (whichever column it landed in) and keep the
        # remaining k neighbors in ascending-distance order
        first_self = (idx == row_idx).float().argmax(dim=1)
        keep_mask  = col_idx != first_self.unsqueeze(1)
        out[b] = idx[keep_mask].reshape(N, k).to(torch.int32)

    return out


def storage_locality_metrics(points: torch.Tensor, half_w: int, sample_queries: int = 4096):
    B, N, _ = points.shape
    adjacent_distance = torch.linalg.vector_norm(
        points[:, 1:, :] - points[:, :-1, :], dim=-1
    ).mean().item()

    sq = min(sample_queries, N)
    query_idx = torch.linspace(0, N - 1, steps=sq, device=points.device).long()
    offsets = torch.arange(-half_w, half_w + 1, device=points.device)
    window_idx = (query_idx[:, None] + offsets[None, :]).clamp(0, N - 1)

    window_distance_sum = 0.0
    for b in range(B):
        q = points[b, query_idx, :].unsqueeze(1)
        cand = points[b, window_idx, :]
        window_distance_sum += torch.linalg.vector_norm(q - cand, dim=-1).mean().item()

    return adjacent_distance, window_distance_sum / B


def timed(fn, iters=20, warmup=3, return_samples=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # one event between each iteration (not one sync per iteration) so we
    # get a per-iteration latency distribution without adding host-sync
    # overhead into the very measurement we're trying to take
    events = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
    events[0].record()
    for i in range(iters):
        fn()
        events[i + 1].record()
    torch.cuda.synchronize()

    samples = [events[i].elapsed_time(events[i + 1]) for i in range(iters)]
    mean = sum(samples) / len(samples)
    if return_samples:
        return mean, samples
    return mean


def compare_one_size(size_label, pt_path, K=K, HALF_W=HALF_W, seed=0):
    torch.manual_seed(seed)

    xs_array = np.load(pt_path)
    xs = torch.from_numpy(xs_array[2]).unsqueeze(0).to(torch.float32).cuda()
    N = xs.shape[1]

    sorted_points, _, _ = morton_sort(xs)

    adj_unsorted, window_unsorted = storage_locality_metrics(xs, HALF_W)
    adj_sorted,   window_sorted   = storage_locality_metrics(sorted_points, HALF_W)

    ms_unsorted = timed(lambda: bruteforce_knn_triton(xs, K, TILE_Q, BF_TILE_T))
    ms_sorted   = timed(lambda: morton_window_knn_triton(xs, K, HALF_W, TILE_Q, TILE_T))

    print(f"  kernel latency   bruteforce/unsorted = {ms_unsorted:.3f} ms   "
          f"morton-windowed/sorted = {ms_sorted:.3f} ms")
    print(f"  adjacent-point distance  unsorted = {adj_unsorted:.4f}   sorted = {adj_sorted:.4f}")
    print(f"  sampled window distance  unsorted = {window_unsorted:.4f}   sorted = {window_sorted:.4f}")

    recall_unsorted = recall_sorted = None
    if N <= RECALL_MAX_N:
        exact = cuml_bruteforce_knn(xs, K)  # independent ground truth (RAPIDS cuML), not either Triton kernel
        approx_unsorted = bruteforce_knn_triton(xs, K, TILE_Q, BF_TILE_T)
        approx_sorted   = morton_window_knn_triton(xs, K, HALF_W, TILE_Q, TILE_T)

        recall_unsorted = recall_at_k(approx_unsorted, exact)
        recall_sorted   = recall_at_k(approx_sorted, exact)
        print(f"  recall vs cuML exact   bruteforce = {recall_unsorted:.4f}   "
              f"morton-windowed = {recall_sorted:.4f}")
    else:
        print(f"  recall skipped (N={N} exceeds RECALL_MAX_N={RECALL_MAX_N}); "
              f"raise the threshold if you have the runtime budget")

    return {
        "size": size_label, "N": N,
        "ms_unsorted": ms_unsorted, "ms_sorted": ms_sorted,
        "recall_unsorted": recall_unsorted,
        "recall_sorted": recall_sorted,
        # "adj_unsorted": adj_unsorted, "adj_sorted": adj_sorted,
        # "window_unsorted": window_unsorted, "window_sorted": window_sorted,
    }


def benchmark_tiling(
    size_label, pt_path,
    K=K, HALF_W=HALF_W, TILE_Q=TILE_Q,
    bf_tile_ts=(32, 64, 128, 256, 512, 1024, 2048, 4096),
    window_tile_ts=(16, 32, 64, 128, 256, 512, 1024),
    iters=30, warmup=5, seed=0,
):
    """Sweep one tile-size knob at a time, all else held fixed, and report
    latency (mean +/- stdev over `iters` samples) and recall against a
    single fixed cuML ground truth computed once per point cloud.

    Two independent knobs are swept, never together:
      - BF_TILE_T only affects `_bruteforce_knn_kernel` (unsorted). Since
        that kernel always scans all N points regardless of chunk size,
        recall here should be ~1.0 and FLAT across BF_TILE_T - only
        latency should move. If recall dips at some tile size, that's a
        chunking/boundary bug, not expected behavior.
      - TILE_T here only affects `_window_knn_kernel` (sorted/morton). It
        only changes how the fixed window of size HALF_W is chunked, not
        which points are visited - so recall should likewise stay flat
        across TILE_T, and any residual recall gap vs. bruteforce is due
        to HALF_W (window radius), not TILE_T.
    """
    torch.manual_seed(seed)
    xs_array = np.load(pt_path)
    xs = torch.from_numpy(xs_array[2]).unsqueeze(0).to(torch.float32).cuda()
    N = xs.shape[1]

    ref = cuml_bruteforce_knn(xs, K) if N <= RECALL_MAX_N else None

    print(f"\n=== tiling sweep: {size_label} (N={N}) ===")
    rows = []

    print(f"-- bruteforce/unsorted: sweeping BF_TILE_T (TILE_Q={TILE_Q} fixed) --")
    for tile_t in bf_tile_ts:
        try:
            fn = lambda tt=tile_t: bruteforce_knn_triton(xs, K, TILE_Q, tt)
            mean_ms, samples = timed(fn, iters=iters, warmup=warmup, return_samples=True)
        except Exception as e:
            print(f"  BF_TILE_T={tile_t:5d}   FAILED: {type(e).__name__}: {e}")
            continue
        std_ms = statistics.stdev(samples)
        recall = recall_at_k(fn(), ref) if ref is not None else None
        recall_str = f"{recall:.4f}" if recall is not None else "n/a"
        print(f"  BF_TILE_T={tile_t:5d}   {mean_ms:8.3f} +/- {std_ms:6.3f} ms   recall={recall_str}")
        rows.append(dict(kernel="bruteforce", size=size_label, N=N, tile_t=tile_t,
                          ms_mean=mean_ms, ms_std=std_ms, recall=recall))

    print(f"-- morton-windowed/sorted: sweeping TILE_T (window-chunk size; HALF_W={HALF_W} fixed) --")
    for tile_t in window_tile_ts:
        try:
            fn = lambda tt=tile_t: morton_window_knn_triton(xs, K, HALF_W, TILE_Q, tt)
            mean_ms, samples = timed(fn, iters=iters, warmup=warmup, return_samples=True)
        except Exception as e:
            print(f"  TILE_T={tile_t:5d}   FAILED: {type(e).__name__}: {e}")
            continue
        std_ms = statistics.stdev(samples)
        recall = recall_at_k(fn(), ref) if ref is not None else None
        recall_str = f"{recall:.4f}" if recall is not None else "n/a"
        print(f"  TILE_T={tile_t:5d}   {mean_ms:8.3f} +/- {std_ms:6.3f} ms   recall={recall_str}")
        rows.append(dict(kernel="morton_windowed", size=size_label, N=N, tile_t=tile_t,
                          ms_mean=mean_ms, ms_std=std_ms, recall=recall))

    return rows


def benchmark_morton_sorted(
    size_label, pt_path,
    K=K, TILE_Q=TILE_Q,
    tile_ts=(16, 32, 64, 128, 256, 512, 1024),
    half_ws=(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    fixed_tile_t=TILE_T,
    fixed_half_w=HALF_W,
    iters=30, warmup=5, seed=0,
):
    """Benchmark ONLY the Morton-sorted windowed Triton kernel, against a
    cuML brute-force reference computed once per point cloud. The naive
    O(N)-per-query Triton bruteforce kernel is never invoked here - cuML
    brute force is fast even at N=1M (~7.5s), so it stands alone as ground
    truth without needing that slow exhaustive kernel.

    Two independent sweeps, each holding the other knob fixed:
      - tile_ts: chunk size used to scan the (fixed-size) window. A pure
        performance knob - the window's contents don't change, so this
        should only move latency, not recall.
      - half_ws: the window radius itself. This is the knob that actually
        trades recall for speed - a bigger window makes more true
        neighbors reachable, at the cost of more work per query.
    """
    torch.manual_seed(seed)
    xs_array = np.load(pt_path)
    xs = torch.from_numpy(xs_array[2]).unsqueeze(0).to(torch.float32).cuda()
    N = xs.shape[1]

    ref = cuml_bruteforce_knn(xs, K)

    print(f"\n=== morton-sorted sweep: {size_label} (N={N}) ===")
    rows = []

    print(f"-- sweeping TILE_T (HALF_W={fixed_half_w} fixed, TILE_Q={TILE_Q}) --")
    for tile_t in tile_ts:
        try:
            fn = lambda tt=tile_t: morton_window_knn_triton(xs, K, fixed_half_w, TILE_Q, tt)
            mean_ms, samples = timed(fn, iters=iters, warmup=warmup, return_samples=True)
        except Exception as e:
            print(f"  TILE_T={tile_t:5d}   FAILED: {type(e).__name__}: {e}")
            continue
        std_ms  = statistics.stdev(samples)
        recall  = recall_at_k(fn(), ref)
        print(f"  TILE_T={tile_t:5d}   {mean_ms:8.3f} +/- {std_ms:6.3f} ms   recall={recall:.4f}")
        rows.append(dict(sweep="tile_t", size=size_label, N=N, tile_t=tile_t, half_w=fixed_half_w,
                          ms_mean=mean_ms, ms_std=std_ms, recall=recall))

    print(f"-- sweeping HALF_W (TILE_T={fixed_tile_t} fixed, TILE_Q={TILE_Q}) --")
    for half_w in half_ws:
        if K > 2 * half_w + 1:
            print(f"  HALF_W={half_w:5d}   SKIPPED: K={K} > 2*HALF_W+1={2 * half_w + 1}")
            continue
        try:
            fn = lambda hw=half_w: morton_window_knn_triton(xs, K, hw, TILE_Q, fixed_tile_t)
            mean_ms, samples = timed(fn, iters=iters, warmup=warmup, return_samples=True)
        except Exception as e:
            print(f"  HALF_W={half_w:5d}   FAILED: {type(e).__name__}: {e}")
            continue
        std_ms  = statistics.stdev(samples)
        recall  = recall_at_k(fn(), ref)
        print(f"  HALF_W={half_w:5d}   {mean_ms:8.3f} +/- {std_ms:6.3f} ms   recall={recall:.4f}")
        rows.append(dict(sweep="half_w", size=size_label, N=N, tile_t=fixed_tile_t, half_w=half_w,
                          ms_mean=mean_ms, ms_std=std_ms, recall=recall))

    return rows


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # results = []
    # for size_label, pt_path in SIZES.items():
    #     results.append(compare_one_size(size_label, pt_path))

    # for res in results:
    #     for k, v in res.items():
    #         print(f"{k}: {v}")

    # cuML-only benchmark: Morton-sorted Triton kernel vs cuML brute force,
    # sweeping TILE_T and HALF_W independently. The naive Triton bruteforce
    # kernel is intentionally never called here (too slow at N=1M).
    sweep_rows = []
    for size_label, pt_path in SIZES.items():
        sweep_rows.extend(benchmark_morton_sorted(size_label, pt_path))

    csv_path = os.path.join(os.path.dirname(__file__), "morton_sorted_tile_halfw_sweep.csv")
    save_csv(sweep_rows, csv_path)
    print(f"\nsaved {len(sweep_rows)} morton-sorted sweep rows to {csv_path}")