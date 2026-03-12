# Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries

## Context

The Transolver-3 paper (arXiv:2602.04940) introduces key optimizations to scale the Physics-Attention mechanism from ~700K cells (Transolver v1) to 160M+ cells for industrial-scale CFD. The current repo implements Transolver v1. This upgrade implements:

1. **Faster Slice & Deslice** — reorder operations via matrix multiplication associativity, moving `Linear1` and `Linear3` from O(N) mesh domain to O(M) slice domain (M << N)
2. **Geometry Slice Tiling** — partition input into tiles processed sequentially with gradient checkpointing, reducing peak memory from O(NM) to O(N_t M)
3. **Geometry Amortized Training** — train on random subsets (100K-400K) of full mesh each iteration
4. **Physical State Caching** — inference: build physical state cache from chunks of the full mesh
5. **Full Mesh Decoding** — inference: decode predictions for any point using cached states

---

## Approach: New `transolver3/` Package

A clean `transolver3/` package is created rather than modifying existing files in-place. Reasons:
- The existing code is duplicated across 4-5 locations
- Preserves backward compatibility and reproducibility of v1 results
- Transolver-3 targets irregular meshes (industrial-scale); structured 2D/3D variants are kept as-is

---

## Architecture

### Phase 1: Core Optimized Physics Attention

**File: `transolver3/__init__.py`** — package init, exports `Transolver3` and `CachedInference`

**File: `transolver3/common.py`** — shared utilities
- `MLP` class (single source of truth, consolidated from duplicated copies)
- `ACTIVATION` dict

**File: `transolver3/physics_attention_v3.py`** — the core innovation

**Class `PhysicsAttentionV3(nn.Module)`**

Architecture changes from `Physics_Attention_Irregular_Mesh` (`Physics_Attention.py:6-58`):

| Component | v1 (current) | v3 (new) |
|-----------|-------------|----------|
| `in_project_x` | `Linear(dim, inner_dim)` — O(NC²) on N points | **REMOVED** from N-domain |
| `in_project_fx` | `Linear(dim, inner_dim)` — O(NC²) on N points | **REMOVED** from N-domain |
| `in_project_slice` | `Linear(dim_head, slice_num)` on `in_project_x(x)` | `Linear(dim, heads * slice_num)` — operates on raw `x` directly |
| (new) `slice_linear1` | — | `Linear(dim, dim_head)` — replaces `in_project_fx` but operates on M slice tokens |
| (new) `slice_linear3` | — | `Linear(dim_head, dim)` + Dropout — replaces `to_out` but operates on M slice tokens |
| `to_out` | `Linear(inner_dim, dim)` — O(NC²) on N points | **REMOVED** from N-domain |
| `to_q`, `to_k`, `to_v` | unchanged | unchanged |
| `temperature` | `nn.Parameter` | `nn.Parameter` (same) |

**Forward — standard path** (Algorithm from paper Eq. 3):
```python
def forward(self, x, num_tiles=0):
    B, N, C = x.shape
    if num_tiles > 1:
        return self.forward_tiled(x, num_tiles)

    # Step 1: Slice weights from raw x — O(NCM)
    w = softmax(self.in_project_slice(x).reshape(B,N,H,M).permute(0,2,1,3) / temperature)  # B,H,N,M
    d = w.sum(dim=2)  # B,H,M

    # Step 2: Aggregate raw x into slice tokens — O(NMC)
    s_raw = einsum("bhnm,bnc->bhmc", w, x)  # B,H,M,C
    s = self.slice_linear1(s_raw / (d[...,None] + 1e-5))  # B,H,M,dim_head — O(MC²) not O(NC²)!

    # Step 3: Self-attention among slices — O(M²C)
    q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
    s_out = F.scaled_dot_product_attention(q, k, v)

    # Step 4: Project out in slice domain — O(MC²) not O(NC²)!
    s_out = self.slice_linear3(s_out)  # B,H,M,C

    # Step 5: Deslice — O(NMC)
    x_out = einsum("bhmc,bhnm->bhnc", s_out, w)  # B,H,N,C
    return rearrange(x_out, 'b h n d -> b n (h d)')
```

**Forward — tiled path** (Algorithm 1 from paper Appendix A.1):
```python
def forward_tiled(self, x, num_tiles):
    B, N, C = x.shape
    tile_size = ceil(N / num_tiles)
    s_raw = zeros(B,H,M,C);  d = zeros(B,H,M)

    for t in range(num_tiles):
        x_t = x[:, t*tile_size:(t+1)*tile_size]
        w_t = checkpoint(compute_weights, x_t)   # B,H,N_t,M
        s_raw += checkpoint(lambda: einsum("bhnm,bnc->bhmc", w_t, x_t))
        d += checkpoint(lambda: w_t.sum(dim=2))

    s = slice_linear1(s_raw / (d[...,None] + 1e-5))
    s_out = slice_linear3(attention(to_q(s), to_k(s), to_v(s)))

    outputs = []
    for t in range(num_tiles):
        x_t = x[:, t*tile_size:(t+1)*tile_size]
        w_t = checkpoint(compute_weights, x_t)  # recomputed
        outputs.append(rearrange(einsum("bhmc,bhnm->bhnc", s_out, w_t), '...'))
    return cat(outputs, dim=1)
```

**Cache methods** (for inference):
- `compute_physical_state(x, num_tiles)` → returns `(s_raw, d)` accumulators
- `compute_cached_state(s_raw, d)` → returns `s_out_cached` (B,H,M,C)
- `decode_from_cache(x_query, s_out_cached)` → computes `w` for query, returns `w * s_out_cached`

### Phase 2: Block and Model

**File: `transolver3/transolver3_block.py`**

**Class `Transolver3Block`** — mirrors `Transolver_block` (`Transolver_Irregular_Mesh.py:40-71`) but uses `PhysicsAttentionV3`:
- `forward(fx, num_tiles=0)` — standard forward with optional tiling
- `compute_physical_state(fx)` — delegates to attention's cache method
- `compute_cached_state(s_raw, d)` — finalize cached state from accumulators
- `forward_from_cache(fx, cached_s_out)` — uses cached states for decoding

**File: `transolver3/model.py`**

**Class `Transolver3(nn.Module)`** — mirrors `Model` (`Transolver_Irregular_Mesh.py:74-158`):
- Same constructor signature + new params: `num_tiles=0`
- Same `preprocess`, `get_grid`, `placeholder`, `initialize_weights`
- `forward(x, fx, T, num_tiles, subset_indices)` — adds amortized training support via `subset_indices`
- `cache_physical_states(x, fx, T, num_tiles, chunk_size)` → `List[cached_s_out]` for all layers
- `decode_from_cache(x_query, cache, fx_query, T)` → predictions using cached states
- `full_mesh_inference(x, fx, T, num_tiles, cache_chunk_size, decode_chunk_size)` → end-to-end two-phase inference

### Phase 3: Training Infrastructure

**File: `transolver3/amortized_training.py`**

- `AmortizedMeshSampler` — generates random subset indices per iteration
- `relative_l2_loss(pred, target)` — paper Eq. 7 loss function
- `create_optimizer(model, lr, weight_decay)` — AdamW per paper config
- `create_scheduler(optimizer, total_steps, warmup_fraction, min_lr)` — cosine LR with linear warmup
- `train_step(model, x, fx, target, optimizer, scheduler, sampler, ...)` — single training step

### Phase 4: Inference Infrastructure

**File: `transolver3/inference.py`**

**Class `CachedInference`**:
- `predict(x, fx, T)` — end-to-end prediction on full mesh of arbitrary size
- `build_cache(x, fx, T)` — Phase 1: partitions full mesh into K chunks that fit in GPU memory, accumulates physical states layer-by-layer, returns `s_cache = {s'_out^(l)}_{l=1}^{L}`
- `decode(x_query, cache, fx_query, T)` — Phase 2: decode predictions for query points using cached states in chunks

### Phase 5: Industrial-Scale Benchmarks

**Directory: `Industrial-Scale-Benchmarks/`**

| File | Purpose |
|------|---------|
| `dataset/nasa_crm.py` | NASA-CRM data loader (~400K surface cells) |
| `dataset/ahmed_ml.py` | AhmedML data loader (~1M surface + ~20M volume) |
| `dataset/drivaer_ml.py` | DrivAerML data loader (~8M surface + ~160M volume) |
| `exp_nasa_crm.py` | Training script: 24 layers, C=256, M=64, full mesh |
| `exp_ahmed_ml.py` | Training script: 16 layers, C=256, M=64, subset 400K |
| `exp_drivaer_ml.py` | Training script: 24 layers, C=256, M=64, subset 400K |
| `utils/metrics.py` | Relative L2 error, R² score, MAE for Cd/Cl, drag/lift coefficients |

Training config (from paper Table 6):
- All: 500 epochs, lr=1e-3, AdamW (weight_decay=0.05), 8 heads, M=64 slices
- NASA-CRM: Full Mesh (small enough), 24 layers
- AhmedML: Amortized 400K subset, 16 layers
- DrivAerML: Amortized 400K subset, 24 layers
- Loss: Relative L2 (`||pred - gt||_2 / ||gt||_2`)
- Cosine LR schedule with 5% warmup, min lr 1e-6
- Gradient clipping at 1.0

### Phase 6: Backward Compatibility

**Modified: `PDE-Solving-StandardBenchmark/model_dict.py`** — added `'Transolver3'` entry pointing to `transolver3.model.Transolver3`

Existing experiments (`exp_darcy.py`, etc.) can use `--model Transolver3` with `num_tiles=0` (no tiling for small benchmarks). The constructor signature is compatible.

---

## File Structure

```
Transolver/
├── transolver3/                          # NEW — core Transolver-3 package
│   ├── __init__.py                       # Exports Transolver3, CachedInference
│   ├── common.py                         # MLP, activations
│   ├── physics_attention_v3.py           # Optimized Physics-Attention (core innovation)
│   ├── transolver3_block.py              # Encoder block with V3 attention
│   ├── model.py                          # Transolver3 model (train + inference)
│   ├── amortized_training.py             # Sampler, loss, optimizer, scheduler
│   └── inference.py                      # CachedInference for industrial-scale
│
├── Industrial-Scale-Benchmarks/          # NEW — industrial benchmark experiments
│   ├── dataset/
│   │   ├── nasa_crm.py                   # NASA-CRM dataset (~400K cells)
│   │   ├── ahmed_ml.py                   # AhmedML dataset (~20M cells)
│   │   └── drivaer_ml.py                 # DrivAerML dataset (~160M cells)
│   ├── utils/
│   │   └── metrics.py                    # L2 error, R², MAE, Cd/Cl computation
│   ├── exp_nasa_crm.py                   # NASA-CRM experiment
│   ├── exp_ahmed_ml.py                   # AhmedML experiment
│   └── exp_drivaer_ml.py                 # DrivAerML experiment
│
├── tests/
│   └── test_transolver3.py               # NEW — 9 tests, all passing
│
├── PDE-Solving-StandardBenchmark/
│   └── model_dict.py                     # MODIFIED — added Transolver3 entry
│
├── Physics_Attention.py                  # UNCHANGED — v1 reference
├── Airfoil-Design-AirfRANS/             # UNCHANGED
├── Car-Design-ShapeNetCar/              # UNCHANGED
└── ...
```

---

## Key Files Reference

| Existing File | Role |
|--------------|------|
| `Physics_Attention.py` | v1 attention — reference for math being optimized |
| `PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py` | v1 block + model — structure replicated |
| `PDE-Solving-StandardBenchmark/model_dict.py` | Registry updated for backward compat |
| `PDE-Solving-StandardBenchmark/exp_airfoil.py` | Training loop pattern followed |

---

## Complexity Analysis

### Original Physics-Attention (v1) — 5 N-related terms

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Linear1(x) | O(NC²) | O(NC) |
| Softmax(Linear2(x)) | O(NCM) | O(NM) |
| (wd⁻¹)ᵀ x_proj | O(NMC) | O(MC) |
| Attention(s) | O(M²C) | O(M² + MC) |
| ws' | O(NMC) | O(NC) |
| Linear3(ws') | O(NC²) | O(NC) |

### Optimized Physics-Attention (v3) — 3 N-related terms

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Softmax(Linear2(x)) | O(NCM) | O(NM) |
| wᵀx | O(NMC) | O(MC) |
| Linear1(s_raw d⁻¹) | O(MC²) | O(MC) |
| Attention(s) | O(M²C) | O(M² + MC) |
| Linear3(s') | O(MC²) | O(MC) |
| ws'_out | O(NMC) | O(NC) |

**Result**: 2 fewer O(N) operations, ~1.9× single-GPU capacity vs Transolver++, ~60% latency reduction.

---

## Verification

All tests pass (`tests/test_transolver3.py`):

1. **Attention shapes** — correct output dimensions (B, N, C)
2. **Tiled vs standard** — numerically equivalent (max diff: 5.96e-08)
3. **Cached vs direct inference** — numerically equivalent (max diff: 0.00)
4. **Block forward** — correct shapes for both intermediate and last layers
5. **Full model forward** — end-to-end with and without tiling
6. **Amortized training** — subset indices produce correct shapes, loss backpropagates
7. **Cached model inference** — full two-phase pipeline produces matching results
8. **Mesh sampler** — correct index generation and edge cases
9. **LR scheduler** — warmup increases, cosine decreases, reaches min_lr

### Additional verification (requires data):
- Run `exp_darcy.py --model Transolver3` and verify convergence comparable to v1
- Profile peak memory with tiling vs without — should show ~T× reduction
- Run NASA-CRM end-to-end with amortized training + cached inference
