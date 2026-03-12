"""
Transolver-3 experiment on NASA Common Research Model (CRM).

NASA-CRM: surface mesh ~400K cells, 105 train + 44 test samples.
This is the smallest industrial benchmark — trains on the full mesh
(no amortized training needed).

Config from Table 6:
  - 500 epochs, lr=1e-3, AdamW, weight_decay=0.05
  - 24 layers, 8 heads, C=256, M=64 slices
  - Full mesh training (no subset)
  - Cosine LR with 5% warmup, min_lr=1e-6
  - Gradient clipping at 1.0
  - Loss: relative L2

Usage:
  python exp_nasa_crm.py --data_dir /path/to/nasa_crm --save_dir ./checkpoints
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transolver3.model import Transolver3
from transolver3.amortized_training import (
    relative_l2_loss, create_optimizer, create_scheduler
)
from transolver3.inference import CachedInference
from Industrial_Scale_Benchmarks.utils.metrics import (
    relative_l2_error, relative_l2_error_per_field
)
from dataset.nasa_crm import NASACRMDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Transolver-3 on NASA-CRM')
    parser.add_argument('--data_dir', required=True, help='Path to NASA-CRM data')
    parser.add_argument('--save_dir', default='./checkpoints/nasa_crm')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=24)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--slice_num', type=int, default=64)
    parser.add_argument('--num_tiles', type=int, default=0,
                        help='Tiles for attention (0=no tiling)')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate for variable-size meshes (batch_size=1 typical)."""
    return {
        key: torch.stack([b[key] for b in batch]) if batch[0][key].dim() > 0
        else torch.stack([b[key] for b in batch])
        for key in batch[0]
    }


def train_epoch(model, dataloader, optimizer, scheduler, args, device):
    model.train()
    total_loss = 0
    count = 0

    for batch in dataloader:
        x = batch['x'].to(device)       # B, N, d_in
        target = batch['target'].to(device)  # B, N, d_out

        optimizer.zero_grad()
        pred = model(x, num_tiles=args.num_tiles)
        loss = relative_l2_loss(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    all_errors = []

    engine = CachedInference(
        model,
        cache_chunk_size=100000,
        decode_chunk_size=50000,
        num_tiles=args.num_tiles,
    )

    for batch in dataloader:
        x = batch['x'].to(device)
        target = batch['target'].to(device)

        # Use cached inference for evaluation
        pred = engine.predict(x.unsqueeze(0) if x.dim() == 2 else x)
        error = relative_l2_error(pred, target)
        all_errors.append(error.cpu())

    mean_error = torch.cat(all_errors).mean().item()
    return mean_error


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_dataset = NASACRMDataset(args.data_dir, split='train')
    test_dataset = NASACRMDataset(args.data_dir, split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, collate_fn=collate_fn)

    # Determine input/output dims from first sample
    sample = train_dataset[0]
    space_dim = sample['x'].shape[-1]
    out_dim = sample['target'].shape[-1]

    # Model
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

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.eval_only:
        error = evaluate(model, test_loader, args, device)
        print(f"Test relative L2 error: {error:.4f} ({error*100:.2f}%)")
        return

    # Training
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = create_scheduler(optimizer, total_steps)

    best_error = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 args, device)
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

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f'model_ep{epoch+1}.pt'))

    print(f"\nBest test relative L2 error: {best_error:.4f} ({best_error*100:.2f}%)")


if __name__ == '__main__':
    main()
