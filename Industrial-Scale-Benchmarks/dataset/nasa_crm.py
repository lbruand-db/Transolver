"""
NASA Common Research Model (CRM) dataset loader.

High-fidelity CFD simulations of a full wing-body-horizontal tail transport
aircraft. Surface mesh with ~400K cells. 6 input parameters: Mach number,
angle of attack, and control surface deflections.

Outputs: surface pressure (p_s) and skin friction coefficient (C_f).

Reference: Bekemeyer et al. (2025), Transolver-3 paper Appendix A.2.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class NASACRMDataset(Dataset):
    """NASA-CRM surface mesh dataset.

    Expected data format (per sample):
        - coords: (N, 3) surface mesh coordinates
        - normals: (N, 3) surface normals
        - inputs: (6,) [Mach, alpha, delta_1, delta_2, delta_3, delta_4]
        - pressure: (N, 1) surface pressure
        - friction: (N, 3) skin friction vectors

    Args:
        data_dir: path to preprocessed data directory
        split: 'train' or 'test'
        normalize_coords: whether to normalize coordinates (min-max scaling)
        coord_scale: scaling factor after normalization (default 1000)
    """

    def __init__(self, data_dir, split='train', normalize_coords=True,
                 coord_scale=1000.0):
        self.data_dir = data_dir
        self.split = split
        self.normalize_coords = normalize_coords
        self.coord_scale = coord_scale

        # Load split file list
        split_file = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.samples = [line.strip() for line in f if line.strip()]
        else:
            # Fall back to listing directory
            self.samples = sorted([
                f for f in os.listdir(data_dir)
                if f.endswith('.npz') or f.endswith('.h5')
            ])

        # Compute normalization stats from training set
        self._target_mean = None
        self._target_std = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(sample_path, allow_pickle=True)

        coords = torch.tensor(data['coords'], dtype=torch.float32)  # N, 3
        normals = torch.tensor(data['normals'], dtype=torch.float32)  # N, 3
        params = torch.tensor(data['inputs'], dtype=torch.float32)  # 6

        # Targets: pressure and friction
        pressure = torch.tensor(data['pressure'], dtype=torch.float32)  # N, 1
        friction = torch.tensor(data['friction'], dtype=torch.float32)  # N, 3
        targets = torch.cat([pressure, friction], dim=-1)  # N, 4

        # Normalize coordinates
        if self.normalize_coords:
            coord_min = coords.min(dim=0, keepdim=True).values
            coord_max = coords.max(dim=0, keepdim=True).values
            coords = (coords - coord_min) / (coord_max - coord_min + 1e-8)
            coords = coords * self.coord_scale

        # Input features: coordinates + normals + condition params broadcast
        N = coords.shape[0]
        params_broadcast = params.unsqueeze(0).expand(N, -1)  # N, 6
        x = torch.cat([coords, normals, params_broadcast], dim=-1)  # N, 12

        return {
            'x': x,           # N, 12 (coords + normals + params)
            'pos': coords,    # N, 3
            'target': targets, # N, 4 (pressure + friction)
            'params': params,  # 6
        }
