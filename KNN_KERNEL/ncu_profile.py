import argparse
import numpy as np
import torch

import triton
import triton.language as tl


SIZES = {
    "4k":  "./blank/blank_pc_4K.npy",
    "16k": "./blank/blank_pc_16K.npy",
    "32k": "./blank/blank_pc_32K.npy",
    "64k": "./blank/blank_pc_64K.npy"
}

K       = 16
HALF_W  = 256
TILE_Q  = 16
TILE_T  = 32       # windowed kernel tile
BF_TILE_T = 256    # bruteforce kernel tile (bigger, since it scans all N)

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
    """Triton kernel: exact brute-force K-nearest-neighbours over a 3D point cloud.

    For each query point (one of TILE_Q per program along axis 0, one
    batch per program along axis 1), scans the entire point set in
    chunks of TILE_T and maintains a running top-K (smallest squared
    distance) via an insertion loop, excluding the point itself.

    Args:
        Pts_ptr: Pointer to the input points buffer, logical shape
            (B, N, 3), float32/float16.
        OutIdx_ptr: Pointer to the output neighbour-index buffer, logical
            shape (B, N, K), int32.
        N: Number of points per batch element (runtime int32 value).
        K (tl.constexpr): Number of nearest neighbours to find per point.
        TILE_Q (tl.constexpr): Number of query points processed per program
            instance along grid axis 0.
        TILE_T (tl.constexpr): Chunk size used while scanning the N points.
        stride_pts_b, stride_pts_n, stride_pts_d: Strides (in elements) of
            `Pts_ptr` along the batch, point, and coordinate dimensions.
        stride_out_b, stride_out_n, stride_out_k: Strides (in elements) of
            `OutIdx_ptr` along the batch, point, and K dimensions.

    Returns:
        None. Writes the K nearest-neighbour indices for each query point
        directly into the buffer at `OutIdx_ptr`.
    """
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
    """Compute exact K-nearest-neighbour indices for a batch of 3D point clouds.

    Launches `_bruteforce_knn_kernel`, which scans the full point set per
    query point (O(N) per query, no spatial pruning) — serves as the
    ground-truth baseline for recall comparisons.

    Args:
        points (torch.Tensor): Input point coordinates, shape (B, N, 3),
            float, CUDA tensor.
        k (int): Number of nearest neighbours to find per point.
        tile_q (int, optional): Query-tile size (points processed per
            program instance). Defaults to 16.
        tile_t (int, optional): Scan chunk size over the N points. Defaults
            to 256.

    Returns:
        torch.Tensor: Nearest-neighbour indices, shape (B, N, k), int32.
    """
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
    """Triton kernel: approximate KNN restricted to a fixed window around each query.

    Assumes points are pre-sorted along a space-filling curve (Morton
    order) so that spatially nearby points are index-adjacent. For each
    query point, only searches a window of `2*HALF_W + TILE_Q` candidates
    centred on the query's own tile, instead of the full N points, then
    keeps a running top-K by squared distance (excluding the query itself).

    Args:
        Pts_ptr: Pointer to the (Morton-)sorted input points buffer, logical
            shape (B, N, 3), float32/float16.
        OutIdx_ptr: Pointer to the output neighbour-index buffer, logical
            shape (B, N, K), int32 (indices are into the sorted array).
        N: Number of points per batch element (runtime int32 value).
        K (tl.constexpr): Number of nearest neighbours to find per point.
        TILE_Q (tl.constexpr): Number of query points processed per program
            instance along grid axis 0.
        TILE_T (tl.constexpr): Chunk size used while scanning the search window.
        HALF_W (tl.constexpr): Half-width of the search window (in points)
            on each side of the query tile.
        stride_pts_b, stride_pts_n, stride_pts_d: Strides (in elements) of
            `Pts_ptr` along the batch, point, and coordinate dimensions.
        stride_out_b, stride_out_n, stride_out_k: Strides (in elements) of
            `OutIdx_ptr` along the batch, point, and K dimensions.

    Returns:
        None. Writes the K nearest-neighbour indices (into the sorted
        array) for each query point directly into the buffer at `OutIdx_ptr`.
    """
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
    """Compute approximate KNN via Morton-order sorting + fixed-window search.

    Sorts the points along a 3D Morton (Z-order) curve so spatially close
    points land at nearby array indices, runs the windowed
    `_window_knn_kernel` over that sorted array (touching far fewer
    candidates than a full scan), then maps the resulting neighbour indices
    back to the original (unsorted) point ordering.

    Args:
        points (torch.Tensor): Input point coordinates, shape (B, N, 3),
            float, CUDA tensor.
        k (int): Number of nearest neighbours to find per point. Must satisfy
            `k <= 2 * half_w + 1`.
        half_w (int, optional): Half-width of the search window (in points)
            on each side of the query tile. Defaults to 256.
        tile_q (int, optional): Query-tile size (points processed per
            program instance). Defaults to 16.
        tile_t (int, optional): Scan chunk size over the search window.
            Defaults to 32.
        morton_bits (int, optional): Bits per dimension used when quantising
            coordinates for the Morton code. Defaults to 16.

    Returns:
        torch.Tensor: Approximate nearest-neighbour indices in the original
            (unsorted) point ordering, shape (B, N, k), int32.
    """
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


# ──morton_sort helper───

def _spread_bits_3(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Spread each bit of an integer with two zero bits, for 3D Morton encoding.

    Transforms x = b(n-1)...b1b0 into ..b(n-1)00b(n-2)00...b1 00 b0, so that
    interleaving three spread values (shifted by 0/1/2) yields a Morton
    (Z-order) code.

    Args:
        x (torch.Tensor): Integer coordinate values to spread, any integer
            dtype tensor (cast internally to int64).
        bits (int): Number of low-order bits of `x` to retain/spread.

    Returns:
        torch.Tensor: Bit-spread values, same shape as `x`, dtype int64.
    """
    x = x.to(torch.int64) & ((1 << bits) - 1)
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8))  & 0x100F00F00F00F00F
    x = (x | (x << 4))  & 0x10C30C30C30C30C3
    x = (x | (x << 2))  & 0x1249249249249249
    return x


def morton_encode_3d(points: torch.Tensor, bits_per_dim: int = 16) -> torch.Tensor:
    """Compute 3D Morton (Z-order) codes for a set of points.

    Normalises point coordinates to the point set's own bounding box,
    quantises each axis to `bits_per_dim` bits, then bit-interleaves the
    three quantised axes into a single integer code such that spatially
    close points tend to have numerically close codes.

    Args:
        points (torch.Tensor): Point coordinates for a single (unbatched)
            point cloud, shape (N, 3), float.
        bits_per_dim (int, optional): Quantisation resolution per axis, in
            bits. Defaults to 16.

    Returns:
        torch.Tensor: Morton codes, shape (N,), dtype int64.
    """
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
    """Sort each point cloud in a batch along its 3D Morton (Z-order) curve.

    Improves spatial locality of nearby points in memory, which the
    windowed KNN kernel relies on to restrict its search to a small
    contiguous range of indices.

    Args:
        points (torch.Tensor): Input point coordinates, shape (B, N, 3), float.
        bits_per_dim (int, optional): Quantisation resolution per axis used
            for the Morton code, in bits. Defaults to 16.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - sorted_points: Points reordered by Morton code, shape
              (B, N, 3), same dtype as `points`.
            - perm: Permutation mapping sorted index -> original index,
              shape (B, N), int64 (`sorted_points[b, i] == points[b, perm[b, i]]`).
            - inv_perm: Inverse permutation mapping original index -> sorted
              index, shape (B, N), int64.
    """
    B, N, _ = points.shape
    perm = torch.empty(B, N, dtype=torch.int64, device=points.device)
    for b in range(B):
        codes = morton_encode_3d(points[b], bits_per_dim=bits_per_dim)
        perm[b] = torch.argsort(codes)
    sorted_points = torch.gather(points, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
    inv_perm = torch.argsort(perm, dim=1)
    return sorted_points, perm, inv_perm


# recall is only affordable up to this size with the O(N) exhaustive kernel
# used as ground truth; above this, recall is skipped and only timing +
# locality metrics are reported. Raise if you have the patience/time budget.
RECALL_MAX_N = 200_000


def recall_at_k(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Compute the fraction of exact neighbours recovered by an approximate KNN result.

    For each query point, checks how many of its approximate neighbour
    indices also appear among its exact neighbour indices (order-independent
    set membership), then averages over all points and batches.

    Args:
        approx (torch.Tensor): Approximate neighbour indices, shape
            (B, N, K), int.
        exact (torch.Tensor): Ground-truth (exact) neighbour indices, shape
            (B, N, K), int.

    Returns:
        float: Mean recall@K in [0, 1] across all query points and batches.
    """
    matches = (approx.sort(-1).values.unsqueeze(-1)
               == exact.sort(-1).values.unsqueeze(-2)).any(-2)
    return matches.float().mean().item()


def storage_locality_metrics(points: torch.Tensor, half_w: int, sample_queries: int = 4096):
    """Measure how spatially local a point ordering is in memory.

    Reports two proxies for cache/memory locality: the mean Euclidean
    distance between index-adjacent points, and the mean distance from a
    sampled set of query points to the candidates within a `+-half_w`
    index window around them (smaller values indicate an ordering where
    nearby-in-memory points tend to be nearby-in-space).

    Args:
        points (torch.Tensor): Point coordinates, shape (B, N, 3), float.
        half_w (int): Half-width of the index window (in points) evaluated
            around each sampled query.
        sample_queries (int, optional): Number of query points to sample
            (evenly spaced) per batch element for the window-distance
            metric. Defaults to 4096.

    Returns:
        Tuple[float, float]:
            - adjacent_distance: Mean distance between consecutive points in
              the given ordering, averaged over the batch.
            - window_distance: Mean distance from sampled queries to
              candidates within their `+-half_w` index window, averaged over
              the batch.
    """
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


def timed(fn, iters=20, warmup=3):
    """Measure the average GPU execution time of a callable using CUDA events.

    Runs `fn` a number of warmup times (untimed, to exclude compilation/
    cache-warming effects), then times `iters` further calls with CUDA
    events and returns the per-call average.

    Args:
        fn (Callable[[], Any]): Zero-argument callable to time; its return
            value is discarded.
        iters (int, optional): Number of timed iterations. Defaults to 20.
        warmup (int, optional): Number of untimed warmup iterations run
            first. Defaults to 3.

    Returns:
        float: Average wall-clock GPU time per call, in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def compare_one_size(size_label, pt_path, K=K, HALF_W=HALF_W, seed=0):
    """Benchmark and compare the bruteforce vs. Morton-windowed KNN kernels on one point cloud.

    Loads a point cloud from disk, times both KNN variants, reports memory-
    locality metrics before/after Morton sorting, and (for point clouds up
    to `RECALL_MAX_N`) computes the windowed kernel's recall@K against the
    exact bruteforce result. Prints a summary to stdout.

    Args:
        size_label (str): Human-readable label for the point-cloud size
            (e.g. "4k"), used only for logging/reporting.
        pt_path (str): Path to a `.npy` file containing point clouds; index
            2 along the first axis is loaded as the sample point cloud.
        K (int, optional): Number of nearest neighbours to find per point.
            Defaults to the module-level `K`.
        HALF_W (int, optional): Half-width of the windowed kernel's search
            window. Defaults to the module-level `HALF_W`.
        seed (int, optional): Random seed set before running. Defaults to 0.

    Returns:
        Dict[str, Any]: Summary metrics with keys "size" (str), "N" (int),
            "ms_unsorted" / "ms_sorted" (float, avg kernel latency in ms),
            "adj_unsorted" / "adj_sorted" (float, adjacent-point distance),
            "window_unsorted" / "window_sorted" (float, sampled window
            distance).
    """
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

    if N <= RECALL_MAX_N:
        exact = bruteforce_knn_triton(xs, K, TILE_Q, BF_TILE_T)  # exhaustive, memory-safe
        approx_unsorted = exact  # same kernel, so recall_unsorted is trivially 1.0 by construction
        approx_sorted   = morton_window_knn_triton(xs, K, HALF_W, TILE_Q, TILE_T)

        recall_sorted = recall_at_k(approx_sorted, exact)
        print(f"  recall vs exact   bruteforce = 1.000 (by definition)   "
              f"morton-windowed = {recall_sorted:.4f}")
    else:
        print(f"  recall skipped (N={N} exceeds RECALL_MAX_N={RECALL_MAX_N}); "
              f"raise the threshold if you have the runtime budget")

    return {
        "size": size_label, "N": N,
        "ms_unsorted": ms_unsorted, "ms_sorted": ms_sorted,
        "adj_unsorted": adj_unsorted, "adj_sorted": adj_sorted,
        "window_unsorted": window_unsorted, "window_sorted": window_sorted,
    }


def profile_one(mode: str, size_label: str, pt_path: str, K=K, HALF_W=HALF_W,
                 seed: int = 0, warmup: int = 3, iters: int = 1):
    """Run a single kernel variant repeatedly so `ncu --launch-skip/--launch-count`
    can isolate one steady-state launch. No timing/recall bookkeeping here -
    this path exists purely to be driven by ncu.

    Args:
        mode (str): Which kernel to run: "unsorted" (bruteforce) or "sorted"
            (Morton-windowed).
        size_label (str): Human-readable size label (unused for logic; kept
            for call-signature symmetry with `compare_one_size`).
        pt_path (str): Path to a `.npy` file containing point clouds; index
            2 along the first axis is loaded as the sample point cloud.
        K (int, optional): Number of nearest neighbours to find per point.
            Defaults to the module-level `K`.
        HALF_W (int, optional): Half-width of the windowed kernel's search
            window (only used when `mode == "sorted"`). Defaults to the
            module-level `HALF_W`.
        seed (int, optional): Random seed set before running. Defaults to 0.
        warmup (int, optional): Untimed kernel launches to run first, so
            `ncu` can skip past them via `--launch-skip`. Defaults to 3.
        iters (int, optional): Kernel launches to run after warmup, for
            `ncu --launch-count` to profile. Defaults to 1.

    Returns:
        None.

    Raises:
        ValueError: If `mode` is not "unsorted" or "sorted".
    """
    torch.manual_seed(seed)

    xs_array = np.load(pt_path)
    xs = torch.from_numpy(xs_array[2]).unsqueeze(0).to(torch.float32).cuda()

    if mode == "unsorted":
        fn = lambda: bruteforce_knn_triton(xs, K, TILE_Q, BF_TILE_T)
    elif mode == "sorted":
        fn = lambda: morton_window_knn_triton(xs, K, HALF_W, TILE_Q, TILE_T)
    else:
        raise ValueError(f"unknown mode: {mode}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZES.keys()), default=None,
                         help="point-cloud size to profile; omit to run the full comparison")
    parser.add_argument("--mode", choices=["sorted", "unsorted"], default=None,
                         help="sorted (morton-windowed) or unsorted (bruteforce) kernel")
    parser.add_argument("--warmup", type=int, default=3,
                         help="kernel launches to skip before the measured one")
    parser.add_argument("--iters", type=int, default=1,
                         help="kernel launches to run after warmup")
    args = parser.parse_args()

    if args.size is not None and args.mode is not None:
        profile_one(args.mode, args.size, SIZES[args.size],
                    warmup=args.warmup, iters=args.iters)
    else:
        results = []
        for size_label, pt_path in SIZES.items():
            results.append(compare_one_size(size_label, pt_path))