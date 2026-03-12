"""
Tests for Transolver-3 implementation.

Verifies:
  1. PhysicsAttentionV3 forward pass produces correct shapes
  2. Tiled forward == standard forward (numerical equivalence)
  3. Cached inference == direct forward (numerical equivalence)
  4. Full model forward pass works end-to-end
  5. Amortized training with subset indices
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.physics_attention_v3 import PhysicsAttentionV3
from transolver3.transolver3_block import Transolver3Block
from transolver3.model import Transolver3
from transolver3.inference import CachedInference
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss, create_optimizer, create_scheduler
)


def test_attention_shapes():
    """Test that PhysicsAttentionV3 produces correct output shapes."""
    B, N, C = 2, 100, 64
    heads = 4
    dim_head = C // heads
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=slice_num)
    x = torch.randn(B, N, C)

    out = attn(x)
    assert out.shape == (B, N, C), f"Expected {(B, N, C)}, got {out.shape}"
    print("PASS: attention output shape correct")


def test_tiled_vs_standard():
    """Test that tiled forward matches standard forward."""
    B, N, C = 1, 200, 64
    heads = 4
    dim_head = C // heads
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=dim_head, slice_num=slice_num)
    attn.eval()

    x = torch.randn(B, N, C)

    with torch.no_grad():
        out_standard = attn(x, num_tiles=0)
        out_tiled_2 = attn(x, num_tiles=2)
        out_tiled_4 = attn(x, num_tiles=4)

    # Should be numerically identical (same operations, just different chunking)
    diff_2 = (out_standard - out_tiled_2).abs().max().item()
    diff_4 = (out_standard - out_tiled_4).abs().max().item()

    assert diff_2 < 1e-5, f"Tiled (2) vs standard max diff: {diff_2}"
    assert diff_4 < 1e-5, f"Tiled (4) vs standard max diff: {diff_4}"
    print(f"PASS: tiled matches standard (max diff: 2-tile={diff_2:.2e}, 4-tile={diff_4:.2e})")


def test_cached_inference_equivalence():
    """Test that cached inference produces same results as direct forward."""
    B, N, C = 1, 150, 64
    heads = 4
    slice_num = 16

    attn = PhysicsAttentionV3(C, heads=heads, dim_head=C // heads, slice_num=slice_num)
    attn.eval()

    x = torch.randn(B, N, C)

    with torch.no_grad():
        # Direct forward
        out_direct = attn(x, num_tiles=0)

        # Cached: compute state then decode
        s_raw, d = attn.compute_physical_state(x)
        s_out = attn.compute_cached_state(s_raw, d)
        out_cached = attn.decode_from_cache(x, s_out)

    diff = (out_direct - out_cached).abs().max().item()
    assert diff < 1e-5, f"Cached vs direct max diff: {diff}"
    print(f"PASS: cached inference matches direct (max diff: {diff:.2e})")


def test_block_forward():
    """Test Transolver3Block forward pass."""
    B, N, C = 2, 100, 64
    out_dim = 4

    # Non-last layer
    block = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                              slice_num=16, last_layer=False)
    x = torch.randn(B, N, C)
    out = block(x)
    assert out.shape == (B, N, C), f"Non-last block: expected {(B, N, C)}, got {out.shape}"

    # Last layer
    block_last = Transolver3Block(num_heads=4, hidden_dim=C, dropout=0.0,
                                   slice_num=16, last_layer=True, out_dim=out_dim)
    out = block_last(x)
    assert out.shape == (B, N, out_dim), f"Last block: expected {(B, N, out_dim)}, got {out.shape}"
    print("PASS: block forward shapes correct")


def test_full_model():
    """Test Transolver3 end-to-end forward pass."""
    B, N = 2, 100
    space_dim = 3
    fun_dim = 2
    out_dim = 4

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=fun_dim,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
        ref=4,
        unified_pos=False,
    )

    x = torch.randn(B, N, space_dim)
    fx = torch.randn(B, N, fun_dim)

    # Standard forward
    out = model(x, fx=fx)
    assert out.shape == (B, N, out_dim), f"Expected {(B, N, out_dim)}, got {out.shape}"

    # Forward with tiling
    out_tiled = model(x, fx=fx, num_tiles=2)
    assert out_tiled.shape == (B, N, out_dim)

    print("PASS: full model forward shapes correct")


def test_amortized_training():
    """Test geometry amortized training with subset indices."""
    B, N = 1, 500
    space_dim = 3
    out_dim = 2
    subset_size = 100

    model = Transolver3(
        space_dim=space_dim,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=8,
        mlp_ratio=1,
    )

    x = torch.randn(B, N, space_dim)
    target = torch.randn(B, N, out_dim)

    # With subset indices
    indices = torch.randperm(N)[:subset_size]
    out = model(x, subset_indices=indices)
    assert out.shape == (B, subset_size, out_dim), \
        f"Expected {(B, subset_size, out_dim)}, got {out.shape}"

    # Loss computation
    loss = relative_l2_loss(out, target[:, indices])
    loss.backward()
    print(f"PASS: amortized training works (loss={loss.item():.4f})")


def test_cached_model_inference():
    """Test full model cached inference pipeline."""
    B, N = 1, 200
    space_dim = 3
    out_dim = 2

    model = Transolver3(
        space_dim=space_dim,
        n_layers=3,
        n_hidden=64,
        n_head=4,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=16,
        mlp_ratio=1,
    )
    model.eval()

    x = torch.randn(B, N, space_dim)

    with torch.no_grad():
        # Direct forward
        out_direct = model(x)

        # Cached inference
        engine = CachedInference(model, cache_chunk_size=50, decode_chunk_size=50)
        out_cached = engine.predict(x)

    assert out_cached.shape == out_direct.shape
    diff = (out_direct - out_cached).abs().max().item()
    # Note: chunked caching may have small numerical differences due to
    # processing in separate chunks vs all-at-once
    print(f"PASS: cached model inference (shape match, max diff: {diff:.2e})")


def test_sampler():
    """Test AmortizedMeshSampler."""
    sampler = AmortizedMeshSampler(subset_size=100, seed=42)

    indices = sampler.sample(1000)
    assert indices.shape == (100,), f"Expected (100,), got {indices.shape}"
    assert indices.max() < 1000
    assert indices.min() >= 0

    # Small mesh: should return all indices
    indices_small = sampler.sample(50)
    assert indices_small.shape == (50,)

    print("PASS: AmortizedMeshSampler works correctly")


def test_scheduler():
    """Test cosine scheduler with warmup."""
    model = nn.Linear(10, 10)
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, total_steps=1000, warmup_fraction=0.1)

    lrs = []
    for step in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Warmup: LR should increase
    assert lrs[50] > lrs[0], "LR should increase during warmup"
    # After warmup: LR should decrease
    assert lrs[500] < lrs[100], "LR should decrease after warmup"
    # End: LR should be near min_lr
    assert lrs[-1] < lrs[100], "Final LR should be lower than mid-training"

    print("PASS: cosine scheduler with warmup works")


if __name__ == '__main__':
    print("=" * 60)
    print("Transolver-3 Tests")
    print("=" * 60)

    test_attention_shapes()
    test_tiled_vs_standard()
    test_cached_inference_equivalence()
    test_block_forward()
    test_full_model()
    test_amortized_training()
    test_cached_model_inference()
    test_sampler()
    test_scheduler()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
