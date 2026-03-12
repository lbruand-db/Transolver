"""
Physical State Caching and Full Mesh Decoding for Transolver-3.

During inference on industrial-scale meshes (>10^8 cells), processing the
full mesh in a single forward pass is memory-prohibitive. Transolver-3
introduces a two-phase decoupled inference:

  Phase 1 - Physical State Caching:
    Partition the full mesh into memory-compatible chunks, accumulate physical
    state contributions (s_raw, d) across chunks for each layer, then compute
    the cached state s'_out per layer.

  Phase 2 - Full Mesh Decoding:
    For each query point (or chunk of query points), decode predictions by
    computing slice weights and multiplying with the cached s'_out. This
    requires only O(1) incremental computation per point per layer.

Reference: Transolver-3 paper, Section 3.3, Figure 3.
"""

import math
import torch


class CachedInference:
    """Manages two-phase inference for industrial-scale meshes.

    Usage:
        engine = CachedInference(model, cache_chunk_size=100000, decode_chunk_size=50000)
        output = engine.predict(x, fx=fx)

    Args:
        model: Transolver3 model instance
        cache_chunk_size: number of mesh points per chunk during state caching.
                         Smaller = less memory but more chunks. Default 100K per paper.
        decode_chunk_size: number of query points decoded per batch during
                          full mesh decoding. Default 50K.
        num_tiles: number of tiles for attention within each chunk (0=no tiling)
    """

    def __init__(self, model, cache_chunk_size=100000, decode_chunk_size=50000,
                 num_tiles=0):
        self.model = model
        self.cache_chunk_size = cache_chunk_size
        self.decode_chunk_size = decode_chunk_size
        self.num_tiles = num_tiles

    @torch.no_grad()
    def predict(self, x, fx=None, T=None):
        """End-to-end prediction on a full mesh of arbitrary size.

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional input features
            T: optional timestep

        Returns:
            output: (B, N, out_dim) predictions
        """
        return self.model.full_mesh_inference(
            x, fx=fx, T=T,
            num_tiles=self.num_tiles,
            cache_chunk_size=self.cache_chunk_size,
            decode_chunk_size=self.decode_chunk_size,
        )

    @torch.no_grad()
    def build_cache(self, x, fx=None, T=None):
        """Build physical state cache (Phase 1 only).

        Useful when you want to decode different query sets against
        the same cached states (e.g., different evaluation resolutions).

        Args:
            x: (B, N, space_dim) full mesh coordinates
            fx: (B, N, fun_dim) optional features
            T: optional timestep

        Returns:
            cache: list of cached states, one per layer
        """
        return self.model.cache_physical_states(
            x, fx=fx, T=T,
            num_tiles=self.num_tiles,
            chunk_size=self.cache_chunk_size,
        )

    @torch.no_grad()
    def decode(self, x_query, cache, fx_query=None, T=None):
        """Decode predictions for query points using existing cache (Phase 2).

        Args:
            x_query: (B, N_q, space_dim) query coordinates
            cache: cached states from build_cache
            fx_query: (B, N_q, fun_dim) optional query features
            T: optional timestep

        Returns:
            output: (B, N_q, out_dim) predictions
        """
        N_q = x_query.shape[1]
        if self.decode_chunk_size is None or self.decode_chunk_size >= N_q:
            return self.model.decode_from_cache(
                x_query, cache, fx_query=fx_query, T=T
            )

        outputs = []
        for start in range(0, N_q, self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, N_q)
            x_q = x_query[:, start:end]
            fx_q = fx_query[:, start:end] if fx_query is not None else None
            out = self.model.decode_from_cache(x_q, cache, fx_query=fx_q, T=T)
            outputs.append(out)

        return torch.cat(outputs, dim=1)
