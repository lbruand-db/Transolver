"""
Optimized Physics-Attention mechanism for Transolver-3.

Key innovations over Transolver v1 (Physics_Attention_Irregular_Mesh):
  1. Faster Slice & Deslice: Linear projections moved from O(N) mesh domain
     to O(M) slice domain via matrix multiplication associativity.
  2. Geometry Slice Tiling: Input partitioned into tiles processed sequentially
     with gradient checkpointing, reducing peak memory from O(NM) to O(N_t*M).
  3. Physical State Caching: Decoupled inference separating state estimation
     from field decoding for industrial-scale meshes.

Reference: Transolver-3 paper (arXiv:2602.04940), Section 3.1-3.3, Algorithm 1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class PhysicsAttentionV3(nn.Module):
    """Optimized Physics-Attention for irregular meshes.

    Compared to v1, this eliminates two O(N)-domain linear projections:
      - in_project_x / in_project_fx are replaced by slice_linear1 operating
        on M slice tokens (M << N).
      - to_out is replaced by slice_linear3 operating on M slice tokens.

    The slice weight computation (in_project_slice) now operates directly on
    the raw input x in dim-space, not on a pre-projected dim_head-space.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        # Learnable temperature for slice weight sharpness
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Slice weight projection: raw x (dim) -> per-head slice weights (heads * slice_num)
        # In v1 this was: in_project_x (dim->inner_dim) then in_project_slice (dim_head->slice_num)
        # In v3: single projection from dim directly to heads*slice_num
        self.in_project_slice = nn.Linear(dim, heads * slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Slice-domain Linear1: replaces in_project_fx but operates on M tokens not N
        # Projects from raw feature dim (dim) to per-head dim (dim_head)
        self.slice_linear1 = nn.Linear(dim, dim_head)

        # Attention Q/K/V projections (unchanged from v1, operate on M slice tokens)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Slice-domain Linear3: replaces to_out but operates on M tokens not N
        # Projects from dim_head back to dim (per head), then we reshape heads*dim -> dim
        # Note: the output per-head is dim-sized to allow proper deslice aggregation
        self.slice_linear3 = nn.Linear(dim_head, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _compute_slice_weights(self, x):
        """Compute slice weights from raw input x.

        Args:
            x: (B, N, C) raw input features

        Returns:
            w: (B, H, N, M) normalized slice weights
        """
        B, N, C = x.shape
        # Project to per-head slice logits
        logits = self.in_project_slice(x)  # B, N, H*M
        logits = logits.reshape(B, N, self.heads, self.slice_num)
        logits = logits.permute(0, 2, 1, 3).contiguous()  # B, H, N, M

        # Temperature-scaled softmax
        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        w = self.softmax(logits / temperature)  # B, H, N, M
        return w

    def forward(self, x, num_tiles=0):
        """Forward pass with optional geometry slice tiling.

        Args:
            x: (B, N, C) input features where N is mesh resolution
            num_tiles: number of tiles for memory-efficient processing.
                       0 or 1 = no tiling (standard path).
                       >1 = tiled processing with gradient checkpointing.

        Returns:
            x_out: (B, N, C) output features
        """
        if num_tiles > 1:
            return self._forward_tiled(x, num_tiles)
        return self._forward_standard(x)

    def _forward_standard(self, x):
        """Standard forward pass (no tiling). Paper Eq. 3."""
        B, N, C = x.shape

        # (1) Faster Slice: compute weights and aggregate in one pass
        # w = Softmax(Linear2(x)) — O(NCM) time, O(NM) space
        w = self._compute_slice_weights(x)  # B, H, N, M
        d = w.sum(dim=2)  # B, H, M — diagonal normalization

        # s_raw = w^T @ x — O(NMC) time, O(MC) space
        # Note: einsum handles B,H,N,M x B,N,C -> B,H,M,C without materializing B,H,N,C
        s_raw = torch.einsum("bhnm,bnc->bhmc", w, x)  # B, H, M, C

        # Linear1 in slice domain — O(MC^2) instead of O(NC^2)
        s = self.slice_linear1(s_raw / (d[..., None] + 1e-5))  # B, H, M, dim_head

        # (2) Self-attention among slice tokens — O(M^2 C)
        q = self.to_q(s)
        k = self.to_k(s)
        v = self.to_v(s)
        s_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )  # B, H, M, dim_head

        # (3) Faster Deslice: Linear3 in slice domain — O(MC^2) instead of O(NC^2)
        s_out = self.slice_linear3(s_out)  # B, H, M, C

        # Deslice: project back to mesh domain — O(NMC)
        # x_out = w @ s_out (using the same slice weights)
        x_out = torch.einsum("bhmc,bhnm->bhnc", s_out, w)  # B, H, N, C

        # Aggregate heads by averaging (since output is already in dim-space per head)
        x_out = x_out.mean(dim=1)  # B, N, C
        return self.out_dropout(x_out)

    def _forward_tiled(self, x, num_tiles):
        """Tiled forward pass with gradient checkpointing. Paper Algorithm 1."""
        B, N, C = x.shape
        tile_size = math.ceil(N / num_tiles)
        device = x.device

        # Initialize global accumulators
        s_raw_accum = torch.zeros(B, self.heads, self.slice_num, C, device=device)
        d_accum = torch.zeros(B, self.heads, self.slice_num, device=device)

        # Phase 1: Accumulate slice contributions from each tile
        for t in range(num_tiles):
            start = t * tile_size
            end = min(start + tile_size, N)
            x_t = x[:, start:end]  # B, N_t, C

            # Gradient checkpoint: recompute w_t during backward pass
            def _tile_slice_and_aggregate(x_tile):
                w_tile = self._compute_slice_weights(x_tile)  # B, H, N_t, M
                s_raw_tile = torch.einsum("bhnm,bnc->bhmc", w_tile, x_tile)  # B, H, M, C
                d_tile = w_tile.sum(dim=2)  # B, H, M
                return s_raw_tile, d_tile

            if self.training:
                s_raw_t, d_t = checkpoint(
                    _tile_slice_and_aggregate, x_t, use_reentrant=False
                )
            else:
                s_raw_t, d_t = _tile_slice_and_aggregate(x_t)

            s_raw_accum = s_raw_accum + s_raw_t
            d_accum = d_accum + d_t

        # Phase 2: Slice-domain operations (on M tokens only)
        s = self.slice_linear1(s_raw_accum / (d_accum[..., None] + 1e-5))  # B, H, M, dim_head

        q = self.to_q(s)
        k = self.to_k(s)
        v = self.to_v(s)
        s_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        s_out = self.slice_linear3(s_out)  # B, H, M, C

        # Phase 3: Deslice per tile
        outputs = []
        for t in range(num_tiles):
            start = t * tile_size
            end = min(start + tile_size, N)
            x_t = x[:, start:end]

            def _tile_deslice(x_tile, s_out_fixed):
                w_tile = self._compute_slice_weights(x_tile)
                x_out_tile = torch.einsum("bhmc,bhnm->bhnc", s_out_fixed, w_tile)
                return x_out_tile.mean(dim=1)  # B, N_t, C

            if self.training:
                x_out_t = checkpoint(
                    _tile_deslice, x_t, s_out, use_reentrant=False
                )
            else:
                x_out_t = _tile_deslice(x_t, s_out)

            outputs.append(x_out_t)

        x_out = torch.cat(outputs, dim=1)  # B, N, C
        return self.out_dropout(x_out)

    # --- Physical State Caching (Inference) ---

    @torch.no_grad()
    def compute_physical_state(self, x, num_tiles=0):
        """Compute cached physical state s'_out from input features.

        Used during the physical state caching phase of inference.
        Accumulates contributions from all chunks/tiles of the full mesh.

        Args:
            x: (B, N, C) input features (possibly a chunk of the full mesh)
            num_tiles: tiling for memory efficiency within this chunk

        Returns:
            s_out: (B, H, M, C) cached physical state after attention
            s_raw: (B, H, M, C) raw aggregated features (for accumulation)
            d: (B, H, M) normalization diagonal (for accumulation)
        """
        B, N, C = x.shape

        if num_tiles > 1:
            tile_size = math.ceil(N / num_tiles)
            s_raw = torch.zeros(B, self.heads, self.slice_num, C, device=x.device)
            d = torch.zeros(B, self.heads, self.slice_num, device=x.device)

            for t in range(num_tiles):
                start = t * tile_size
                end = min(start + tile_size, N)
                x_t = x[:, start:end]
                w_t = self._compute_slice_weights(x_t)
                s_raw = s_raw + torch.einsum("bhnm,bnc->bhmc", w_t, x_t)
                d = d + w_t.sum(dim=2)
        else:
            w = self._compute_slice_weights(x)
            s_raw = torch.einsum("bhnm,bnc->bhmc", w, x)
            d = w.sum(dim=2)

        return s_raw, d

    @torch.no_grad()
    def compute_cached_state(self, s_raw, d):
        """From accumulated s_raw and d, compute the final cached state.

        Args:
            s_raw: (B, H, M, C) accumulated raw slice features
            d: (B, H, M) accumulated normalization

        Returns:
            s_out: (B, H, M, C) the cached physical state
        """
        s = self.slice_linear1(s_raw / (d[..., None] + 1e-5))
        q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
        s_out = F.scaled_dot_product_attention(q, k, v)
        s_out = self.slice_linear3(s_out)  # B, H, M, C
        return s_out

    @torch.no_grad()
    def decode_from_cache(self, x_query, cached_s_out):
        """Decode predictions for query points using cached physical states.

        Paper Eq. 5: w^(l) = Softmax(Linear2(x^(l))), x_out = w * s'_out

        Args:
            x_query: (B, N_q, C) query point features
            cached_s_out: (B, H, M, C) cached physical state from compute_cached_state

        Returns:
            x_out: (B, N_q, C) decoded output
        """
        w = self._compute_slice_weights(x_query)  # B, H, N_q, M
        x_out = torch.einsum("bhmc,bhnm->bhnc", cached_s_out, w)  # B, H, N_q, C
        x_out = x_out.mean(dim=1)  # B, N_q, C
        return x_out
