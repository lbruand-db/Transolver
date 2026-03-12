"""
Transolver-3 experiment on DrivAerML.

DrivAerML: ~160M cells per case, 500 samples (400 train, 50 val, 50 test).
The largest industrial benchmark. Surface: ~8.8M points per sample.
Volume: ~140M points per sample.

Config from Table 6:
  - 500 epochs, lr=1e-3, AdamW, weight_decay=0.05
  - 24 layers, 8 heads, C=256, M=64 slices
  - Subset size: 400K (amortized training)
  - Cosine LR with 5% warmup, min_lr=1e-6

Usage:
  python exp_drivaer_ml.py --data_dir /path/to/drivaer_ml --field surface
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    AmortizedMeshSampler, relative_l2_loss,
    create_optimizer, create_scheduler,
)
from transolver3.inference import CachedInference
from Industrial_Scale_Benchmarks.utils.metrics import (
    relative_l2_error, relative_l2_error_per_field
)
from dataset.drivaer_ml import DrivAerMLDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Transolver-3 on DrivAerML')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', default='./checkpoints/drivaer_ml')
    parser.add_argument('--field', default='surface', choices=['surface', 'volume', 'both'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--subset_size', type=int, default=400000)
    parser.add_argument('--n_layers', type=int, default=24)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=64)
    parser.add_argument('--num_tiles', type=int, default=8,
                        help='More tiles for this large benchmark')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--cache_chunk_size', type=int, default=100000,
                        help='Chunk size for physical state caching during eval')
    parser.add_argument('--decode_chunk_size', type=int, default=50000,
                        help='Chunk size for decoding during eval')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def get_field_key(field):
    return f'{field}_x', f'{field}_target'


def train_epoch(model, dataloader, optimizer, scheduler, sampler, args, device):
    model.train()
    total_loss = 0
    count = 0
    x_key, t_key = get_field_key(args.field)

    for batch in dataloader:
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        optimizer.zero_grad()

        N = x.shape[1]
        indices = sampler.sample(N).to(device)
        x_sub = x[:, indices]
        target_sub = target[:, indices]

        pred = model(x_sub, num_tiles=args.num_tiles)
        loss = relative_l2_loss(pred, target_sub)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    """Evaluate using full cached inference pipeline."""
    model.eval()
    all_errors = []
    x_key, t_key = get_field_key(args.field)

    engine = CachedInference(
        model,
        cache_chunk_size=args.cache_chunk_size,
        decode_chunk_size=args.decode_chunk_size,
        num_tiles=args.num_tiles,
    )

    for batch in dataloader:
        x = batch[x_key].to(device)
        target = batch[t_key].to(device)

        # Two-phase cached inference for industrial-scale mesh
        cache = engine.build_cache(x)
        pred = engine.decode(x, cache)

        error = relative_l2_error(pred, target)
        all_errors.append(error.cpu())

        # Per-field errors
        field_errors = relative_l2_error_per_field(pred, target)
        field_names = ['p_s', 'tau_x', 'tau_y', 'tau_z'] if args.field == 'surface' \
            else ['u_x', 'u_y', 'u_z', 'p_v']
        for name, err in zip(field_names[:len(field_errors)], field_errors):
            print(f"  {name}: {err:.4f} ({err*100:.2f}%)")

    return torch.cat(all_errors).mean().item()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_dataset = DrivAerMLDataset(args.data_dir, split='train', field=args.field,
                                     subset_size=args.subset_size)
    test_dataset = DrivAerMLDataset(args.data_dir, split='test', field=args.field,
                                    subset_size=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sample = train_dataset[0]
    x_key, t_key = get_field_key(args.field)
    space_dim = sample[x_key].shape[-1]
    out_dim = sample[t_key].shape[-1]

    model = Transolver3(
        space_dim=space_dim,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,
        out_dim=out_dim,
        slice_num=args.slice_num,
        mlp_ratio=1,
        dropout=0.0,
        num_tiles=args.num_tiles,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training subset size: {args.subset_size:,}")
    print(f"Tiles: {args.num_tiles}, Cache chunks: {args.cache_chunk_size:,}")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.eval_only:
        error = evaluate(model, test_loader, args, device)
        print(f"Test relative L2 error: {error:.4f} ({error*100:.2f}%)")
        return

    sampler = AmortizedMeshSampler(args.subset_size)
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, total_steps)

    best_error = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 sampler, args, device)
        t1 = time.time()

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_error = evaluate(model, test_loader, args, device)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"test_L2={test_error:.4f} ({test_error*100:.2f}%) | "
                  f"time={t1-t0:.1f}s")
            if test_error < best_error:
                best_error = test_error
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, 'best_model.pt'))
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.6f} | time={t1-t0:.1f}s")

    print(f"\nBest test relative L2 error: {best_error:.4f} ({best_error*100:.2f}%)")


if __name__ == '__main__':
    main()
