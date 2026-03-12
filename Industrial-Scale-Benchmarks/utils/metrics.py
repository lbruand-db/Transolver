"""
Evaluation metrics for industrial-scale benchmarks.

Reference: Transolver-3 paper, Appendix A.3.
"""

import torch
import numpy as np


def relative_l2_error(pred, target):
    """Per-sample relative L2 error (paper Eq. 7).

    L2_rel = ||pred - target||_2 / ||target||_2

    Args:
        pred: (B, N, d_out) or (N, d_out) predictions
        target: same shape as pred

    Returns:
        (B,) or scalar relative L2 error per sample
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    diff_norm = torch.norm(pred - target, p=2, dim=(1, 2))
    target_norm = torch.norm(target, p=2, dim=(1, 2))
    return diff_norm / (target_norm + 1e-8)


def relative_l2_error_per_field(pred, target):
    """Per-field relative L2 error.

    Computes separate relative L2 for each output dimension.

    Args:
        pred: (B, N, d_out)
        target: (B, N, d_out)

    Returns:
        (d_out,) mean relative L2 per field
    """
    d_out = pred.shape[-1]
    errors = []
    for d in range(d_out):
        diff_norm = torch.norm(pred[..., d] - target[..., d], p=2, dim=-1)
        target_norm = torch.norm(target[..., d], p=2, dim=-1)
        errors.append((diff_norm / (target_norm + 1e-8)).mean().item())
    return errors


def r_squared(y_true, y_pred):
    """R-squared score for coefficient predictions (paper Eq. 11).

    Args:
        y_true: (M,) ground truth values
        y_pred: (M,) predicted values

    Returns:
        R^2 score (scalar)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def mean_absolute_error(y_true, y_pred):
    """MAE for integrated coefficients (paper Eq. 12).

    Args:
        y_true: (M,) ground truth
        y_pred: (M,) predictions

    Returns:
        MAE (scalar)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return np.mean(np.abs(y_true - y_pred))


def compute_drag_lift_coefficients(pressure, wall_shear_stress, normals, areas,
                                   rho_inf, v_inf, ref_area,
                                   drag_direction, lift_direction, p_inf=0.0):
    """Compute aerodynamic drag and lift coefficients (paper Eq. 9-10).

    F = integral_S [-(p - p_inf) * n + tau] dS
    C_d = F . d_hat / (0.5 * rho * v^2 * A)
    C_l = F . l_hat / (0.5 * rho * v^2 * A)

    Args:
        pressure: (N_s,) or (N_s, 1) surface pressure
        wall_shear_stress: (N_s, 3) wall shear stress vectors
        normals: (N_s, 3) outward unit normals
        areas: (N_s,) cell areas
        rho_inf: freestream density
        v_inf: freestream velocity magnitude
        ref_area: reference area
        drag_direction: (3,) unit vector for drag direction
        lift_direction: (3,) unit vector for lift direction
        p_inf: freestream reference pressure (default 0)

    Returns:
        C_d: drag coefficient (scalar)
        C_l: lift coefficient (scalar)
    """
    if isinstance(pressure, torch.Tensor):
        pressure = pressure.cpu().numpy()
    if isinstance(wall_shear_stress, torch.Tensor):
        wall_shear_stress = wall_shear_stress.cpu().numpy()
    if isinstance(normals, torch.Tensor):
        normals = normals.cpu().numpy()
    if isinstance(areas, torch.Tensor):
        areas = areas.cpu().numpy()

    pressure = pressure.flatten()
    # Force = integral [-(p - p_inf) * n + tau] dS
    # Discretized: F = sum_i [-(p_i - p_inf) * n_i + tau_i] * A_i
    pressure_force = -(pressure - p_inf)[:, None] * normals * areas[:, None]
    shear_force = wall_shear_stress * areas[:, None]
    total_force = (pressure_force + shear_force).sum(axis=0)  # (3,)

    dynamic_pressure = 0.5 * rho_inf * v_inf ** 2 * ref_area
    C_d = np.dot(total_force, drag_direction) / dynamic_pressure
    C_l = np.dot(total_force, lift_direction) / dynamic_pressure

    return C_d, C_l
