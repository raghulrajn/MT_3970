"""Model– Contact-Aware Tool Encoding with FiLM Conditioning.

Pipeline
--------
    blank0 + tools@T1 + params  ->  stage1  ->  blank1_hat
    blank1_in + blank0 + tools@T2 + params  ->  stage2  ->  blank2_hat
    blank2_in + blank0 + params  ->  springback  ->  blank3_hat

Expected batch structure
------------------------
    batch["blank"]      : (B, 4, N, 3)
    batch["die"]        : (B, 4, Md, 3)   Md = 2048
    batch["punch"]      : (B, 4, Mp, 3)   Mp = 2048
    batch["binder"]     : (B, 4, Mb, 3)   Mb = 512
    batch["parameters"] : (B, 5)          process params[:, 3:]
"""

# from pykeops.torch import LazyTensor
import math
import re
from typing import Any, Dict, Optional, Tuple

from numpy import rec
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseSurrogateModel
torch._dynamo.config.capture_scalar_outputs = True

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbour features without an N×N expansion.

    Args:
        points : (B, N, C)
        idx    : (B, N, K) long indices into dim-1

    Returns:
        (B, N, K, C)
    """
    B, N, C = points.shape
    _, _, K = idx.shape
    B_idx = (
        torch.arange(B, device=points.device)
        .view(B, 1, 1).expand(B, N, K)
    )
    return points[B_idx, idx, :]

def knn_indices(feat:torch.Tensor, k:int)->torch.Tensor:
    """Find, for every point, the indices of its k nearest neighbours (self-excluded).

    Computes pairwise squared Euclidean distances within a single point set
    and takes the k smallest (excluding the point itself, which has distance 0).

    Args:
        feat (torch.Tensor): Point features/coordinates, shape (B, N, C), float.
        k (int): Number of neighbours to return per point.

    Returns:
        torch.Tensor: Long tensor of neighbour indices, shape (B, N, k),
            values in [0, N-1].
    """
    B, N, C = feat.shape
    x2 = (feat * feat).sum(dim=-1, keepdim=True)
    dist2 = x2 + x2.transpose(-1,-2) - 2.0 * torch.bmm(feat, feat.transpose(-1,-2))
    dist2 = dist2.clamp(min=0.0)
    idx = dist2.topk(k + 1, largest=False, dim=-1).indices[:, :, 1:]
    return idx.clamp(0, N - 1)


def _cross_knn(blank_xyz: torch.Tensor, tool_xyz: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find, for each point in one set, the K nearest neighbours in another set.

    Computes pairwise squared Euclidean distances between two (possibly
    different-sized) point sets and returns the K smallest per query point.

    Args:
        blank_xyz (torch.Tensor): Query point coordinates, shape (B, N, 3), float.
        tool_xyz (torch.Tensor): Reference point coordinates to search over,
            shape (B, M, 3), float.
        K (int): Number of neighbours to return per query point.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - distances: Euclidean distances to the K neighbours, shape (B, N, K), float.
            - indices: Long tensor of neighbour indices into `tool_xyz`, shape (B, N, K).
    """
    bx2 = (blank_xyz * blank_xyz).sum(-1, keepdim=True)
    tx2 = (tool_xyz  * tool_xyz ).sum(-1, keepdim=True)
    dist2 = bx2 + tx2.transpose(-1,-2) - 2.0 * torch.bmm(
        blank_xyz, tool_xyz.transpose(-1,-2)
    )
    dist2 = dist2.clamp(min=0.0)
    knn_dists_sq, knn_idx = dist2.topk(K, largest=False, dim=-1)  
    knn_idx = knn_idx.contiguous() 
    return knn_dists_sq.clamp(min=0.0).sqrt(), knn_idx

class _ToolEdgeConv(nn.Module):
    """Single EdgeConv layer operating on a tool point cloud.

    Builds the tool-internal KNN graph in current feature space, computes
    edge features (f_i || f_j - f_i), and reduces with max-pool over K
    neighbours (canonical DGCNN aggregation, Wang et al. 2019).
    """

    def __init__(self, in_ch: int, out_ch: int, k: int):
        """Initialise the layer's KNN size and edge-feature MLP.

        Args:
            in_ch (int): Number of input feature channels per point.
            out_ch (int): Number of output feature channels per point.
            k (int): Number of nearest neighbours used to build the graph.
        """
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 2, out_ch),
            nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply one EdgeConv graph-convolution pass.

        Args:
            feat (torch.Tensor): Input point features, shape (B, M, in_ch), float.

        Returns:
            torch.Tensor: Updated point features, shape (B, M, out_ch), float.
        """
        k = self.k
        knn_dists, knn_idx = _cross_knn(feat, feat, k)
        nbr  = index_points(feat, knn_idx)  
        feat_i = feat.unsqueeze(2).expand(-1, -1, k, -1)
        edge_delta = nbr - feat_i
        edge = torch.cat([feat_i.contiguous(), edge_delta.contiguous()], dim=-1)
        return self.mlp(edge).amax(dim=2)                        # (B, M, out_ch)


class LocalToolEncoder(nn.Module):
    
    def __init__(self, tool_dim: int, k_tool: int = 16):
        """Initialise the two-layer EdgeConv stack used to embed a tool point cloud.

        Args:
            tool_dim (int): Output feature dimension of the tool embedding.
            k_tool (int, optional): Number of nearest neighbours used by each
                EdgeConv layer. Defaults to 16.
        """
        super().__init__()
        self.ec1 = _ToolEdgeConv(3,   64,       k_tool)
        self.ec2 = _ToolEdgeConv(64,  tool_dim, k_tool)

    def forward(self, tool_xyz: torch.Tensor) -> torch.Tensor:
        """Encode a tool point cloud into per-point local features.

        Args:
            tool_xyz (torch.Tensor): Normalised tool coordinates, shape (B, M, 3), float.

        Returns:
            torch.Tensor: Per-point tool features, shape (B, M, tool_dim), float.
        """
        f1 = self.ec1(tool_xyz)          # (B, M, 64)
        f2 = self.ec2(f1)                # (B, M, tool_dim)
        return f2


class ContactAwareToolAttention(nn.Module):

    def __init__(self, tool_dim: int, K_cross: int = 16, n_freq: int = 8):
        """Initialise the cross-attention projections and distance-bias MLP.

        Args:
            tool_dim (int): Feature dimension of the tool embeddings and of
                the attention query/key/value/output.
            K_cross (int, optional): Number of nearest tool points attended to
                per blank point. Defaults to 16.
            n_freq (int, optional): Number of sinusoidal frequency bands used
                to encode neighbour distance. Defaults to 8.
        """
        super().__init__()
        self.K_cross = K_cross
        self.n_freq  = n_freq
        self.dist_bias_mlp = nn.Sequential(
            nn.Linear(2 * n_freq, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # Projection matrices for Q (from blank XYZ), K and V (from tool feat)
        self.proj_q = nn.Linear(3,        tool_dim, bias=False)
        self.proj_k = nn.Linear(tool_dim, tool_dim, bias=False)
        self.proj_v = nn.Linear(tool_dim, tool_dim, bias=False)
        self.scale  = tool_dim ** -0.5
        self.out_norm = nn.LayerNorm(tool_dim)

    # --------------------------------------------------

    def _sinusoidal_dist_enc(self, d: torch.Tensor) -> torch.Tensor:
        """Encode scalar distances as multi-frequency sine/cosine features.

        Args:
            d (torch.Tensor): Neighbour distances, shape (B, N, K), float.

        Returns:
            torch.Tensor: Sinusoidal distance encoding, shape
                (B, N, K, 2 * n_freq), float.
        """
        freqs  = 2.0 ** torch.arange(self.n_freq, device=d.device, dtype=d.dtype)                                                        # (n_freq,)
        angles = d.unsqueeze(-1) * freqs * math.pi              # (B, N, K, n_freq)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # --------------------------------------------------

    def forward(
        self,
        blank_xyz:  torch.Tensor,   # (B, N, 3)
        tool_xyz:   torch.Tensor,   # (B, M, 3)
        tool_feat:  torch.Tensor,   # (B, M, tool_dim)
    ) -> torch.Tensor:
        """Compute per-blank-point contact-aware context from a tool point cloud.

        For every blank point, gathers its K nearest tool points and runs
        scaled dot-product attention over them, biased by a learned function
        of the (sinusoidally encoded) blank-to-tool distance.

        Args:
            blank_xyz (torch.Tensor): Blank point coordinates, shape (B, N, 3), float.
            tool_xyz (torch.Tensor): Tool point coordinates, shape (B, M, 3), float.
            tool_feat (torch.Tensor): Per-point tool features, shape
                (B, M, tool_dim), float.

        Returns:
            torch.Tensor: Per-blank-point tool context features, shape
                (B, N, tool_dim), float.
        """
        B, N, _ = blank_xyz.shape
        M = tool_xyz.shape[1]
        
        K = self.K_cross
        knn_dists, knn_idx = _cross_knn(blank_xyz, tool_xyz, K)          # (B, N, K) each
        tool_dim_  = tool_feat.shape[2]
        idx_flat = knn_idx.flatten(start_dim=1, end_dim=2)
        idx_flat_c = idx_flat.unsqueeze(-1).expand(B, N * K, tool_dim_)
        nbr_feat   = torch.gather(tool_feat, dim=1, index=idx_flat_c).reshape(B, N, K, tool_dim_)
        # --- Scaled dot-product attention + distance bias --------------------
        q     = self.proj_q(blank_xyz).unsqueeze(2)             # (B, N, 1, tool_dim)
        k_mat = self.proj_k(nbr_feat)                           # (B, N, K, tool_dim)
        v_mat = self.proj_v(nbr_feat)                           # (B, N, K, tool_dim)
        # # Distance bias: log-frequency sinusoidal encoding -> MLP -> scalar
        dist_enc   = self._sinusoidal_dist_enc(knn_dists)       # (B, N, K, 2*n_freq)
        dist_bias  = self.dist_bias_mlp(dist_enc).squeeze(-1)   # (B, N, K)
        B, N, _, tool_dim = q.shape

        q_flat = q.view(B * N, 1, tool_dim)
        k_flat = k_mat.view(B * N, K, tool_dim)
        v_flat = v_mat.view(B * N, K, tool_dim)
        bias_flat = dist_bias.view(B * N, 1, K)
        context_flat = F.scaled_dot_product_attention(query=q_flat,key=k_flat,value=v_flat,attn_mask=bias_flat,dropout_p=0.0)
        context = context_flat.view(B, N, tool_dim)

        return self.out_norm(context)

class FiLMGenerator(nn.Module):

    def __init__(
        self,
        raw_param_dim: int,
        feature_dim:   int,
        hidden_dim:    int   = 128,
        dropout:       float = 0.1,
    ):
        """Initialise the MLP that maps raw process parameters to FiLM (gamma, beta).

        The final linear layer is zero-initialised so the module starts as an
        identity transform (gamma=1, beta=0).

        Args:
            raw_param_dim (int): Dimension of the raw input process-parameter vector.
            feature_dim (int): Dimension of the feature map to be modulated;
                also the size of each of gamma and beta.
            hidden_dim (int, optional): Width of the hidden layers. Defaults to 128.
            dropout (float, optional): Dropout probability applied after each
                hidden layer. Defaults to 0.1.
        """
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(raw_param_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim * 2),   # outputs [gamma_offset | beta]
        )

        # Identity initialisation: gamma = 1 + 0 = 1,  beta = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute FiLM modulation parameters from raw process parameters.

        Args:
            params (torch.Tensor): Raw process parameters, shape
                (B, raw_param_dim), float.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - gamma: Multiplicative modulation, shape (B, feature_dim), float.
                - beta: Additive modulation, shape (B, feature_dim), float.
        """
        out           = self.net(params) # (B, 2*feature_dim)
        gamma_off, beta = out.chunk(2, dim=-1)
        gamma         = 1.0 + gamma_off # residual = identity
        return gamma, beta

class EdgeConv(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        knn_chunk_size: Optional[int] = None,
    ):
        """Initialise the layer's KNN size and edge-feature MLP.

        Args:
            in_ch (int): Number of input feature channels per point.
            out_ch (int): Number of output feature channels per point.
            k (int): Number of nearest neighbours used to build the graph.
            knn_chunk_size (Optional[int], optional): Unused chunk-size hint
                reserved for memory-bounded KNN computation. Defaults to None.
        """
        super().__init__()
        self.k              = k
        self.knn_chunk_size = knn_chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 2, out_ch),
            nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply one EdgeConv graph-convolution pass over a self-KNN graph.

        Args:
            feat (torch.Tensor): Input point features, shape (B, N, in_ch), float.

        Returns:
            torch.Tensor: Updated point features, shape (B, N, out_ch), float.
        """
        idx = knn_indices(feat, self.k)
        nbr = index_points(feat, idx)
        K = nbr.shape[2]
        fi = feat.unsqueeze(2).expand(-1, -1, K, -1)
        edge_delta = nbr - fi
        edge = torch.cat([fi.contiguous(), edge_delta.contiguous()], dim=-1)
        return self.mlp(edge).amax(dim=2)

class DGCNNBackbone(nn.Module):

    def __init__(
        self,
        k: int,
        in_ch: int,
        hidden_dims: Tuple[int, int, int],
        fuse_dim: int,
        residual: bool = True,
        knn_chunk_size: Optional[int] = None,
        k_growth: Tuple[int, int, int] = (1, 2, 3),
    ):
        """
        Args:
            k          : base K for EdgeConv layer 1.
            k_growth   : per-layer multipliers applied to k.
                         Default (1, 2, 3) → k, 2k, 3k.
                         Set (1, 1, 1) to reproduce the original fixed-K behaviour.
            hidden_dims: output channels for each of the three EdgeConv blocks.
            fuse_dim   : output channels of the skip-concat MLP.
        """
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.residual = residual

        k1 = max(1, k * k_growth[0])
        k2 = max(1, k * k_growth[1])
        k3 = max(1, k * k_growth[2])

        self.ec1  = EdgeConv(in_ch, h1, k1, knn_chunk_size)
        self.ec2  = EdgeConv(h1,    h2, k2, knn_chunk_size)
        self.ec3  = EdgeConv(h2,    h3, k3, knn_chunk_size)
        self.fuse = nn.Sequential(
            nn.Linear(h1 + h2 + h3, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three-layer EdgeConv stack and fuse multi-scale features.

        Args:
            x (torch.Tensor): Input point features, shape (B, N, in_ch), float.

        Returns:
            torch.Tensor: Fused multi-scale point features, shape
                (B, N, fuse_dim), float.
        """
        y1 = self.ec1(x)
        y2 = self.ec2(y1)
        if self.residual and y2.shape == y1.shape:
            y2 = y2 + y1
        y3 = self.ec3(y2)
        if self.residual and y3.shape == y2.shape:
            y3 = y3 + y2
        return self.fuse(torch.cat([y1, y2, y3], dim=-1))

class ContactAwareStageNet(nn.Module):

    def __init__(
        self,
        k:              int,
        tool_dim:       int,
        hidden_dims:    Tuple[int, int, int],
        fuse_dim:       int,
        residual:       bool,
        knn_chunk_size: Optional[int],
        K_cross:        int,
        n_freq:         int,
        raw_param_dim:  int,
        param_hidden:   int,
        film_dropout:   float,
        k_growth:       Tuple[int, int, int] = (1, 2, 3),
    ):
        """Initialise one forming stage: tool cross-attention + DGCNN + FiLM + delta head.

        Args:
            k (int): Base K for the DGCNN backbone's first EdgeConv layer.
            tool_dim (int): Feature dimension of the encoded tool point clouds.
            hidden_dims (Tuple[int, int, int]): Output channels of the three
                DGCNN EdgeConv blocks.
            fuse_dim (int): Output channels of the DGCNN skip-concat fusion MLP.
            residual (bool): Whether to add residual connections between
                consecutive EdgeConv layers in the DGCNN backbone.
            knn_chunk_size (Optional[int]): Chunk-size hint forwarded to the
                DGCNN backbone's EdgeConv layers.
            K_cross (int): Number of nearest tool points attended to per
                blank point in each cross-attention module.
            n_freq (int): Number of sinusoidal frequency bands used to encode
                cross-attention neighbour distance.
            raw_param_dim (int): Dimension of the raw process-parameter vector.
            param_hidden (int): Hidden width of the FiLM generator MLP.
            film_dropout (float): Dropout probability inside the FiLM generator.
            k_growth (Tuple[int, int, int], optional): Per-layer multipliers
                applied to `k` in the DGCNN backbone. Defaults to (1, 2, 3).
        """
        super().__init__()

        # --- Contact-aware cross-attention for each tool ---------------------
        self.attn_die    = ContactAwareToolAttention(tool_dim, K_cross, n_freq)
        self.attn_punch  = ContactAwareToolAttention(tool_dim, K_cross, n_freq)
        self.attn_binder = ContactAwareToolAttention(tool_dim, K_cross, n_freq)

        # --- DGCNN backbone with growing-K receptive field -------------------
        # in_ch: blank_curr(3) + blank_ref(3) + die_ctx + punch_ctx + binder_ctx
        in_ch = 6 + 3 * tool_dim
        self.backbone = DGCNNBackbone(
            k=k, in_ch=in_ch, hidden_dims=hidden_dims,
            fuse_dim=fuse_dim, residual=residual,
            knn_chunk_size=knn_chunk_size,
            k_growth=k_growth,
        )

        # --- FiLM generator --------------------------------------------------
        self.film_gen = FiLMGenerator(
            raw_param_dim=raw_param_dim,
            feature_dim=fuse_dim,
            hidden_dim=param_hidden,
            dropout=film_dropout,
        )

        # --- Delta head (Kaiming uniform init by PyTorch default) ------------
        self.head_delta = nn.Linear(fuse_dim, 3)

    def forward(
        self,
        blank_curr:   torch.Tensor,   # (B, N, 3)
        blank_ref:    torch.Tensor,
        die_xyz:      torch.Tensor,   # (B, Md, 3)
        die_feat:     torch.Tensor,   # (B, Md, tool_dim)
        punch_xyz:    torch.Tensor,   # (B, Mp, 3)
        punch_feat:   torch.Tensor,   # (B, Mp, tool_dim)
        binder_xyz:   torch.Tensor,   # (B, Mb, 3)
        binder_feat:  torch.Tensor,   # (B, Mb, tool_dim)
        params:       torch.Tensor,   # (B, P)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the next blank shape from the current blank, tools, and process params.

        Runs contact-aware cross-attention against each tool, fuses the
        results with the current/reference blank through a DGCNN backbone,
        applies FiLM conditioning on the process parameters, and predicts a
        residual displacement.

        Args:
            blank_curr (torch.Tensor): Current blank point coordinates, shape
                (B, N, 3), float.
            blank_ref (torch.Tensor): Reference blank point coordinates
                (e.g. initial blank), shape (B, N, 3), float.
            die_xyz (torch.Tensor): Die tool coordinates, shape (B, Md, 3), float.
            die_feat (torch.Tensor): Die tool per-point features, shape
                (B, Md, tool_dim), float.
            punch_xyz (torch.Tensor): Punch tool coordinates, shape (B, Mp, 3), float.
            punch_feat (torch.Tensor): Punch tool per-point features, shape
                (B, Mp, tool_dim), float.
            binder_xyz (torch.Tensor): Binder tool coordinates, shape (B, Mb, 3), float.
            binder_feat (torch.Tensor): Binder tool per-point features, shape
                (B, Mb, tool_dim), float.
            params (torch.Tensor): Raw process parameters, shape (B, P), float.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - blank_next: Predicted next blank point coordinates, shape
                  (B, N, 3), float.
                - delta: Predicted per-point displacement, shape (B, N, 3), float.
        """
        die_ctx    = self.attn_die(blank_curr, die_xyz, die_feat)       # (B, N, tool_dim)
        punch_ctx  = self.attn_punch(blank_curr, punch_xyz, punch_feat) # (B, N, tool_dim)
        binder_ctx = self.attn_binder(blank_curr, binder_xyz, binder_feat)
    
    
        x = torch.cat([blank_curr, blank_ref, die_ctx, punch_ctx, binder_ctx], dim=-1)
        f = self.backbone(x)                                             # (B, N, fuse_dim)
        gamma, beta = self.film_gen(params)                              # (B, fuse_dim) each
        f = gamma.unsqueeze(1) * f + beta.unsqueeze(1)                  # (B, N, fuse_dim)

        # 5. Delta prediction and residual update
        delta      = self.head_delta(f)                                  # (B, N, 3)
        blank_next = blank_curr + delta
        return (blank_next, delta)

class FiLMSpringbackNet(nn.Module):

    def __init__(
        self,
        k:              int,
        hidden_dims:    Tuple[int, int, int],
        fuse_dim:       int,
        residual:       bool,
        knn_chunk_size: Optional[int],
        raw_param_dim:  int,
        param_hidden:   int,
        film_dropout:   float,
        k_growth:       Tuple[int, int, int] = (1, 2, 3),
    ):
        """Initialise the springback stage: DGCNN + FiLM + delta head (no tool contact).

        Args:
            k (int): Base K for the DGCNN backbone's first EdgeConv layer.
            hidden_dims (Tuple[int, int, int]): Output channels of the three
                DGCNN EdgeConv blocks.
            fuse_dim (int): Output channels of the DGCNN skip-concat fusion MLP.
            residual (bool): Whether to add residual connections between
                consecutive EdgeConv layers in the DGCNN backbone.
            knn_chunk_size (Optional[int]): Chunk-size hint forwarded to the
                DGCNN backbone's EdgeConv layers.
            raw_param_dim (int): Dimension of the raw process-parameter vector.
            param_hidden (int): Hidden width of the FiLM generator MLP.
            film_dropout (float): Dropout probability inside the FiLM generator.
            k_growth (Tuple[int, int, int], optional): Per-layer multipliers
                applied to `k` in the DGCNN backbone. Defaults to (1, 2, 3).
        """
        super().__init__()
        in_ch = 6   # blank2_in (3) + blank0 (3)

        self.backbone = DGCNNBackbone(
            k=k, in_ch=in_ch, hidden_dims=hidden_dims,
            fuse_dim=fuse_dim, residual=residual,
            knn_chunk_size=knn_chunk_size,
            k_growth=k_growth,
        )
        self.film_gen = FiLMGenerator(
            raw_param_dim=raw_param_dim,
            feature_dim=fuse_dim,
            hidden_dim=param_hidden,
            dropout=film_dropout,
        )
        self.head_delta = nn.Linear(fuse_dim, 3)

    def forward(
        self,
        blank2:  torch.Tensor,   # (B, N, 3)
        blank0:  torch.Tensor,   # (B, N, 3)
        params:  torch.Tensor,   # (B, P)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the sprung-back final blank shape from the post-forming blank.

        Args:
            blank2 (torch.Tensor): Post-forming blank point coordinates
                (stage-2 output, before springback), shape (B, N, 3), float.
            blank0 (torch.Tensor): Initial blank point coordinates, shape
                (B, N, 3), float.
            params (torch.Tensor): Raw process parameters, shape (B, P), float.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - blank3: Predicted final (sprung-back) blank point
                  coordinates, shape (B, N, 3), float.
                - delta: Predicted per-point springback displacement, shape
                  (B, N, 3), float.
        """
        x = torch.cat([blank2, blank0], dim=-1)                          # (B, N, 6)
        f = self.backbone(x)                                             # (B, N, fuse_dim)

        gamma, beta = self.film_gen(params)
        f = gamma.unsqueeze(1) * f + beta.unsqueeze(1)

        delta      = self.head_delta(f)                                  # (B, N, 3)
        blank3     = blank2 + delta
        return blank3, delta

class Model_ContactAware(BaseSurrogateModel):
    blank1_hat: Optional[torch.Tensor] #type of all the tensors should be mentioned for model compilation
    blank2_hat: Optional[torch.Tensor]
    blank3_hat: Optional[torch.Tensor]
    d01_hat: Optional[torch.Tensor]
    d12_hat: Optional[torch.Tensor]
    d23_hat: Optional[torch.Tensor]
    _gt_blank1: Optional[torch.Tensor]
    _gt_blank2: Optional[torch.Tensor]
    
    def __init__(self, config: Dict[str, Any]):
        """Build the full three-stage contact-aware sheet-metal-forming surrogate model.

        Reads architecture, loss-weight, scheduled-sampling, and
        noise-injection settings from `config` and constructs the tool
        encoders, the two contact-aware forming stages, and the springback
        stage.

        Args:
            config (Dict[str, Any]): Model configuration dict. Recognised
                top-level keys: "architecture", "loss_weights",
                "scheduled_sampling", "noise_injection" (see BaseSurrogateModel
                for any other base config).
        """
        super().__init__(config)

        arch = config.get("architecture", {})

        # Architecture hyper-parameters
        self.tool_dim        = arch.get("tool_embed_dim",    32)
        self.raw_param_dim   = arch.get("param_dim",          5)
        self.k               = arch.get("knn_k",             10)
        self.k_tool          = arch.get("k_tool",            16)
        self.K_cross         = arch.get("K_cross",           16)
        self.n_freq          = arch.get("n_freq",             8)
        self.hidden_dims     = tuple(arch.get("hidden_dims",  [128, 128, 128]))

        self.hidden_dims_s2  = tuple(arch.get("hidden_dims_stage2",
                                              arch.get("hidden_dims", [128, 128, 128])))
        self.fuse_dim        = arch.get("fuse_dim",          128)
        self.residual        = arch.get("residual",          True)
        self.knn_chunk_size  = arch.get("knn_chunk_size",    None)
        self.param_hidden    = arch.get("param_hidden_dim",  128)
        self.film_dropout    = arch.get("film_dropout",      0.10)

        self.k_growth        = tuple(arch.get("k_growth",    [1, 2, 3]))

        # Loss weights
        lw = config.get("loss_weights", {})
        self.w_b1  = float(lw.get("w_blank1", 1.0))
        self.w_b2  = float(lw.get("w_blank2", 1.0))
        self.w_b3  = float(lw.get("w_blank3", 1.0))
        self.w_d01 = float(lw.get("w_d01",    0.0))
        self.w_d12 = float(lw.get("w_d12",    0.0))
        self.w_d23 = float(lw.get("w_d23",    0.0))
        self.w_lap = float(lw.get("w_lap",    0.0))
        self.lap_k = int(lw.get("lap_k",      12))
        
        self.w_lap_s1 = float(lw.get("w_lap_stage1", 0.0))
        self.w_lap_s2 = float(lw.get("w_lap_stage2", 0.0))
        
        self.focal_gamma = float(lw.get("focal_gamma", 0.0))

        ss = config.get("scheduled_sampling", {})
        self.ss_ramp_epochs = int(ss.get("ramp_epochs",  8))
        self.ss_detach_pred = bool(ss.get("detach_pred", True))
        self.ss_start_alpha = float(ss.get("start_alpha", 1.0))
        self.ss_end_alpha   = float(ss.get("end_alpha",   0.0))

        noise = config.get("noise_injection", {})
        self.noise_ramp_epochs = int(noise.get("ramp_epochs",  0))
        self.noise_std_start   = float(noise.get("std_start",  0.0))
        self.noise_std_end     = float(noise.get("std_end",    0.0))
        self._epoch: int = 0

        self.tool_enc_die    = LocalToolEncoder(self.tool_dim, self.k_tool)
        self.tool_enc_punch  = LocalToolEncoder(self.tool_dim, self.k_tool)
        self.tool_enc_binder = LocalToolEncoder(self.tool_dim, self.k_tool)

        _stage_kw = dict(
            k              = self.k,
            tool_dim       = self.tool_dim,
            hidden_dims    = self.hidden_dims,
            fuse_dim       = self.fuse_dim,
            residual       = self.residual,
            knn_chunk_size = self.knn_chunk_size,
            K_cross        = self.K_cross,
            n_freq         = self.n_freq,
            raw_param_dim  = self.raw_param_dim,
            param_hidden   = self.param_hidden,
            film_dropout   = self.film_dropout,
            k_growth       = self.k_growth,
        )
    
        _stage2_kw = {**_stage_kw, "hidden_dims": self.hidden_dims_s2}

        self.stage1     = ContactAwareStageNet(**_stage_kw)
        self.stage2     = ContactAwareStageNet(**_stage2_kw)

        self.springback = FiLMSpringbackNet(
            k              = self.k,
            hidden_dims    = self.hidden_dims,
            fuse_dim       = self.fuse_dim,
            residual       = self.residual,
            knn_chunk_size = self.knn_chunk_size,
            raw_param_dim  = self.raw_param_dim,
            param_hidden   = self.param_hidden,
            film_dropout   = self.film_dropout,
            k_growth       = self.k_growth,
        )

        self.blank1_hat: Optional[torch.Tensor] = None
        self.blank2_hat: Optional[torch.Tensor] = None
        self.blank3_hat: Optional[torch.Tensor] = None
        self.d01_hat:    Optional[torch.Tensor] = None
        self.d12_hat:    Optional[torch.Tensor] = None
        self.d23_hat:    Optional[torch.Tensor] = None

    def set_epoch(self, epoch: int) -> None:
        """Record the current training epoch, used to schedule sampling/noise ramps.

        Args:
            epoch (int): Current epoch number (0-indexed).

        Returns:
            None
        """
        self._epoch = int(epoch)

    @staticmethod
    def _linear_ramp(epoch: int, ramp_epochs: int, start: float, end: float) -> float:
        """Linearly interpolate a scalar from `start` to `end` over `ramp_epochs`.

        Args:
            epoch (int): Current epoch number.
            ramp_epochs (int): Number of epochs over which to ramp. If <= 0,
                the ramp is skipped and `end` is returned immediately.
            start (float): Value at epoch 0.
            end (float): Value once `epoch >= ramp_epochs`.

        Returns:
            float: Interpolated value, clamped to the [start, end] range
                (in ramp order).
        """
        if ramp_epochs <= 0:
            return end
        t = min(max(epoch / float(ramp_epochs), 0.0), 1.0)
        return start + t * (end - start)

    def _scheduled_sampling_alpha(self, epoch: int) -> float:
        """Compute the current teacher-forcing probability for scheduled sampling.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Probability alpha in [0, 1] of using ground truth (vs. the
                model's own prediction) as input to the next stage.
        """
        return self._linear_ramp(
            epoch, self.ss_ramp_epochs, self.ss_start_alpha, self.ss_end_alpha
        )

    def _noise_std(self, epoch: int) -> float:
        """Compute the current standard deviation for input noise injection.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Standard deviation of noise to inject, ramped between
                `noise_std_start` and `noise_std_end`.
        """
        return self._linear_ramp(
            epoch, self.noise_ramp_epochs, self.noise_std_start, self.noise_std_end
        )

    @staticmethod
    def _mix_gt_pred(gt, pred, alpha, detach_pred):
        """Stochastic per-sample Bernoulli scheduled sampling (Bengio et al.
        NeurIPS 2015, §2.4).  Soft blending is deliberately avoided.

        Args:
            gt (torch.Tensor): Ground-truth blank tensor, shape (B, N, 3), float.
            pred (torch.Tensor): Model-predicted blank tensor, shape (B, N, 3), float.
            alpha (float): Per-batch probability of selecting `gt` over `pred`
                for a given sample.
            detach_pred (bool): If True, detach `pred` from the autograd graph
                before mixing (prevents gradient flow through the sampling
                choice for the prediction branch).

        Returns:
            torch.Tensor: Per-sample mix of `gt` and `pred`, shape (B, N, 3), float.
        """
        pred_b = pred.detach() if detach_pred else pred
        B      = gt.shape[0]
        mask   = (torch.rand(B, device=gt.device) < alpha).float().view(B, 1, 1)
        return mask * gt + (1.0 - mask) * pred_b

    def _laplacian_smoothness(self, field: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Compute a graph-Laplacian smoothness penalty on a per-point field.

        Penalises deviation of each point's value from the mean of its
        precomputed neighbours, encouraging spatially smooth displacement fields.

        Args:
            field (torch.Tensor): Per-point field to regularise (e.g. a
                predicted displacement), shape (B, N, C), float.
            idx (torch.Tensor): Precomputed neighbour indices, shape (B, N, K), long.

        Returns:
            torch.Tensor: Scalar mean squared Laplacian, shape (), float.
        """
        nbr = index_points(field, idx)           # (B, N, K, C)
        lap = field - nbr.mean(dim=2)            # (B, N, C)
        return (lap ** 2).mean()

    @staticmethod
    def _focal_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Compute a focal-weighted mean squared error over per-point predictions.

        Points whose squared error exceeds the per-sample median are
        up-weighted by `(1 + gamma)`, focusing training on harder points.

        Args:
            pred (torch.Tensor): Predicted point coordinates, shape (B, N, 3), float.
            target (torch.Tensor): Ground-truth point coordinates, shape
                (B, N, 3), float.
            gamma (float): Focal up-weighting strength for hard (above-median
                error) points. If 0.0, reduces to plain MSE.

        Returns:
            torch.Tensor: Scalar (focal) mean squared error, shape (), float.
        """
        sq_err = (pred - target).pow(2).sum(dim=-1)          # (B, N)
        if gamma == 0.0:
            return sq_err.mean()
        with torch.no_grad():
            median_err = sq_err.median(dim=1, keepdim=True).values  # (B, 1)
            hard_mask  = (sq_err > median_err).float()              # (B, N)
            weights    = 1.0 + gamma * hard_mask                    # (B, N)
        return (weights * sq_err).mean()

    # -------------------------------------------------- data preparation ---

    def preprocess_data(self, batch: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Unpack a raw batch dict into model inputs and the training target.

        Extracts the initial blank, the die/punch/binder tool clouds at
        timesteps T1 and T2, and the process parameters. If ground-truth
        blank states at all 4 timesteps are present, caches the intermediate
        ground-truth blanks and displacement deltas (used later by
        `compute_loss`) and sets the target to the final blank; otherwise
        falls back to the initial blank as a placeholder target (inference mode).

        Args:
            batch (Any): Dict-like batch with keys:
                - "blank": (B, T, N, 3) blank point coordinates over T timesteps
                  (T >= 4 for training, else only blank0 is used).
                - "die": (B, T, Md, 3) die tool coordinates.
                - "punch": (B, T, Mp, 3) punch tool coordinates.
                - "binder": (B, T, Mb, 3) binder tool coordinates.
                - "parameters": (B, P) raw process parameters.

        Returns:
            Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
                - inputs: 8-tuple
                  (blank0, die@T1, punch@T1, binder@T1, die@T2, punch@T2,
                  binder@T2, params), each a contiguous float tensor, ready
                  to pass to `forward`.
                - target: Ground-truth final blank coordinates (or blank0 if
                  no ground truth is available), shape (B, N, 3), float.
        """
        blank  = batch["blank"]
        die    = batch["die"]
        punch  = batch["punch"]
        binder = batch["binder"]
        params = batch["parameters"]

        blank0 = blank[:, 0]
        self._blank0_ref = blank0

        if blank.shape[1] >= 4:
            self._gt_blank1 = blank[:, 1]
            self._gt_blank2 = blank[:, 2]
            self._gt_blank3 = blank[:, 3]
            self._gt_d01    = blank[:, 1] - blank[:, 0]
            self._gt_d12    = blank[:, 2] - blank[:, 1]
            self._gt_d23    = blank[:, 3] - blank[:, 2]
            target = blank[:, 3]
        else:
            self._gt_blank1 = None
            self._gt_blank2 = None
            self._gt_blank3 = None
            self._gt_d01    = None
            self._gt_d12    = None
            self._gt_d23    = None
            target = blank0

        inputs = (
        blank0.contiguous(),
        die[:, 1].contiguous(),   punch[:, 1].contiguous(),   binder[:, 1].contiguous(),
        die[:, 2].contiguous(),   punch[:, 2].contiguous(),   binder[:, 2].contiguous(),
        params.contiguous(),
    )
        return inputs, target

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run the full 3-stage forming pipeline: stage1 -> stage2 -> springback.

        Encodes the die/punch/binder tool clouds at both timesteps, predicts
        blank1 from blank0 (stage1), optionally scheduled-samples between the
        prediction and ground truth as input to stage2, predicts blank2, does
        the same scheduled-sampling mix for the springback stage, and finally
        predicts blank3. Ground-truth references are only used when
        `self.training` is True, and intermediate predictions are cached as
        `self.blank1_hat`, `self.blank2_hat`, `self.blank3_hat`,
        `self.d01_hat`, `self.d12_hat`, `self.d23_hat` for `compute_loss`.

        Args:
            inputs (Tuple[torch.Tensor, ...]): 8-tuple as produced by
                `preprocess_data`:
                (blank0, die1, punch1, binder1, die2, punch2, binder2, params).
                blank0: (B, N, 3); die1/die2: (B, Md, 3); punch1/punch2:
                (B, Mp, 3); binder1/binder2: (B, Mb, 3); params: (B, P).
                All float tensors.

        Returns:
            torch.Tensor: Predicted final (sprung-back) blank point
                coordinates, shape (B, N, 3), float.
        """
        (blank0,
         die1, punch1, binder1,
         die2, punch2, binder2,
         params) = inputs

        epoch     = getattr(self, "current_epoch", self._epoch)
        alpha     = self._scheduled_sampling_alpha(epoch)
        noise_std = self._noise_std(epoch)

        # zero out GT references so no downstream code can
        # accidentally leak ground truth during validation / test passes.
        gt_blank1 = getattr(self, "_gt_blank1", None) if self.training else None
        gt_blank2 = getattr(self, "_gt_blank2", None) if self.training else None

        # ---- Pre-encode per-point tool features (geometry, no blank needed) --
        # Shared encoders used for both timesteps.
        die1_feat    = self.tool_enc_die(die1)        # (B, Md, tool_dim)
        punch1_feat  = self.tool_enc_punch(punch1)    # (B, Mp, tool_dim)
        binder1_feat = self.tool_enc_binder(binder1)  # (B, Mb, tool_dim)

        die2_feat    = self.tool_enc_die(die2)
        punch2_feat  = self.tool_enc_punch(punch2)
        binder2_feat = self.tool_enc_binder(binder2)

        # ---- Stage 1: blank0 + tools@T1 → blank1_hat -----------------------
        blank1_hat, d01_hat = self.stage1(
            blank0,blank0,
            die1,    die1_feat,
            punch1,  punch1_feat,
            binder1, binder1_feat,
            params,
        )
        self.blank1_hat, self.d01_hat = blank1_hat, d01_hat

        if gt_blank1 is not None and self.ss_ramp_epochs > 0:
            blank1_in = self._mix_gt_pred(
                gt_blank1, blank1_hat, alpha, self.ss_detach_pred
            )
        else:
            blank1_in = blank1_hat

        # ---- Stage 2: blank1_in + tools@T2 → blank2_hat --------------------
        blank2_hat, d12_hat = self.stage2(
            blank1_in, blank0,
            die2,    die2_feat,
            punch2,  punch2_feat,
            binder2, binder2_feat,
            params,
        )
        self.blank2_hat, self.d12_hat = blank2_hat, d12_hat

        if gt_blank2 is not None and self.ss_ramp_epochs > 0:
            blank2_in = self._mix_gt_pred(
                gt_blank2, blank2_hat, alpha, self.ss_detach_pred
            )
        else:
            blank2_in = blank2_hat

        # ---- Springback: blank2_in + blank0 + params → blank3_hat ----------
        blank3_hat, d23_hat = self.springback(blank2_in, blank0, params)
        self.blank3_hat, self.d23_hat = blank3_hat, d23_hat

        return blank3_hat

    # ----------------------------------------------------------- losses ---

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total weighted training loss over all stages.

        Combines (focal) MSE position losses for blank1/blank2/blank3,
        optional MSE losses on the per-stage displacement deltas, and
        optional Laplacian smoothness penalties on those deltas, each scaled
        by its configured weight (`self.w_b1`, `self.w_b2`, `self.w_b3`,
        `self.w_d01`, `self.w_d12`, `self.w_d23`, `self.w_lap_s1`/`self.w_lap`,
        `self.w_lap_s2`/`self.w_lap`, `self.w_lap`). Must be called after
        `forward()` since it reads cached predictions set there.

        Args:
            predictions (torch.Tensor): Predicted final blank coordinates
                (model's `forward` output), shape (B, N, 3), float.
            targets (torch.Tensor): Ground-truth final blank coordinates,
                shape (B, N, 3), float.

        Returns:
            torch.Tensor: Scalar total training loss, shape (), float.
        """
        assert self.blank3_hat is not None, "forward() must precede compute_loss()"

        g = self.focal_gamma
        dev = predictions.device
        # --- Primary blank position losses -----------------------------------
        loss_b3 = self._focal_mse(predictions, targets, g)

        loss_b1 = torch.tensor(0.0, device=dev)
        loss_b2 = torch.tensor(0.0, device=dev)
        if self._gt_blank1 is not None and self.blank1_hat is not None and self.w_b1 > 0:
            loss_b1 = self._focal_mse(self.blank1_hat, self._gt_blank1, g)
        if self._gt_blank2 is not None and self.blank2_hat is not None and self.w_b2 > 0:
            loss_b2 = self._focal_mse(self.blank2_hat, self._gt_blank2, g)

        # --- Delta displacement losses ---------------------------------------
        loss_d01 = loss_d12 = loss_d23 = predictions.new_zeros(())
        if self.w_d01 > 0 and self._gt_d01 is not None and self.d01_hat is not None:
            loss_d01 = F.mse_loss(self.d01_hat, self._gt_d01)
        if self.w_d12 > 0 and self._gt_d12 is not None and self.d12_hat is not None:
            loss_d12 = F.mse_loss(self.d12_hat, self._gt_d12)
        if self.w_d23 > 0 and self._gt_d23 is not None and self.d23_hat is not None:
            loss_d23 = F.mse_loss(self.d23_hat, self._gt_d23)

        loss_lap_s1 = loss_lap_s2 = loss_lap = predictions.new_zeros(())

        _w_s1 = self.w_lap_s1 if self.w_lap_s1 > 0 else self.w_lap
        _w_s2 = self.w_lap_s2 if self.w_lap_s2 > 0 else self.w_lap

        any_lap = (_w_s1 > 0 or _w_s2 > 0 or self.w_lap > 0)
        if any_lap:
            # Build KNN on blank0 (exact, no prediction error)
            blank0_ref = (
                self._gt_blank1 - self._gt_d01
                if (self._gt_blank1 is not None and self._gt_d01 is not None)
                else predictions.new_zeros(predictions.shape)
            )
            # Prefer to use the cached blank0 from preprocess_data if available
            if hasattr(self, "_blank0_ref"):
                blank0_ref = self._blank0_ref
            lap_idx_0 = knn_indices(blank0_ref.detach(),
                                        self.lap_k)

            if _w_s1 > 0 and self.d01_hat is not None:
                loss_lap_s1 = self._laplacian_smoothness(self.d01_hat, lap_idx_0)

            if _w_s2 > 0 and self.d12_hat is not None:
                loss_lap_s2 = self._laplacian_smoothness(self.d12_hat, lap_idx_0)

            if self.w_lap > 0 and self.d23_hat is not None:
                xyz_ref = (
                    self.blank2_hat.detach()
                    if self.blank2_hat is not None
                    else self._gt_blank2
                )
                lap_idx_2 = knn_indices(xyz_ref.detach(),
                                            self.lap_k)
                loss_lap = self._laplacian_smoothness(self.d23_hat, lap_idx_2)

        return (
            self.w_b1  * loss_b1
            + self.w_b2  * loss_b2
            + self.w_b3  * loss_b3
            + self.w_d01 * loss_d01
            + self.w_d12 * loss_d12
            + self.w_d23 * loss_d23
            + _w_s1      * loss_lap_s1
            + _w_s2      * loss_lap_s2
            + self.w_lap * loss_lap
        )

    def configure_optimizers(self):
        """Build the optimizer specified by `self.config["optimizer"]`.

        Reads "name" ("adamw", "adam", or "sgd"), "learning_rate",
        "weight_decay", "betas" (for adamw/adam), and "momentum" (for sgd)
        from the optimizer config sub-dict.

        Returns:
            torch.optim.Optimizer: A configured AdamW, Adam, or SGD optimizer
                over `self.parameters()`.

        Raises:
            ValueError: If `name` is not one of "adamw", "adam", "sgd".
        """
        opt_cfg = self.config.get("optimizer", {})
        name    = opt_cfg.get("name",          "adamw").lower()
        lr      = opt_cfg.get("learning_rate",  1e-3)
        wd      = opt_cfg.get("weight_decay",   1e-4)
        betas   = tuple(opt_cfg.get("betas",    [0.9, 0.999]))

        if name == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=wd, betas=betas
            )
        if name == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=wd, betas=betas
            )
        if name == "sgd":
            momentum = opt_cfg.get("momentum", 0.9)
            return torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=wd, momentum=momentum
            )
        raise ValueError(f"Unsupported optimizer: {name}")