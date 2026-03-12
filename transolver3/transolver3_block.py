import torch.nn as nn
from transolver3.common import MLP
from transolver3.physics_attention_v3 import PhysicsAttentionV3


class Transolver3Block(nn.Module):
    """Transolver-3 encoder block with optimized Physics-Attention."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionV3(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, num_tiles=0):
        fx = self.Attn(self.ln_1(fx), num_tiles=num_tiles) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

    def compute_physical_state(self, fx):
        """Compute raw physical state for caching (inference).

        Returns (s_raw, d) that can be accumulated across mesh chunks.
        """
        return self.Attn.compute_physical_state(self.ln_1(fx))

    def compute_cached_state(self, s_raw, d):
        """Finalize cached state from accumulated (s_raw, d)."""
        return self.Attn.compute_cached_state(s_raw, d)

    def forward_from_cache(self, fx, cached_s_out):
        """Forward using pre-computed physical states (inference decoding).

        Paper Eq. 5: uses cached s'_out instead of recomputing attention.
        """
        fx = self.Attn.decode_from_cache(self.ln_1(fx), cached_s_out) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx
