from __future__ import annotations

import torch

from manifold_opt.geometry.types import Point, Tangent, validate_point, validate_tangent


def _project_to_ball(y: torch.Tensor, x: torch.Tensor, r: float) -> torch.Tensor:
    """
    Project y onto the Euclidean ball centered at x with radius r.

    The radius must be non-negative. Inputs are expected to be finite tensors.
    """
    if r < 0:
        raise ValueError("Radius r must be non-negative.")
    if not torch.isfinite(y).all() or not torch.isfinite(x).all():
        raise ValueError("_project_to_ball inputs must contain only finite values.")

    d = y - x
    norm_d = torch.norm(d, dim=-1, keepdim=True)
    safe_norm = torch.clamp(norm_d, min=torch.finfo(y.dtype).tiny)
    scale = torch.where(norm_d <= r, torch.ones_like(norm_d), r / safe_norm)

    projected = x + scale * d
    if not torch.isfinite(projected).all():
        raise ValueError("_project_to_ball produced non-finite values.")
    return projected


def feasible_set_projection(point: Point, tangent: Tangent, r: float) -> Tangent:
    """
    Project tangent = (xi_U, y, xi_V) onto C^r_point.

    For point = (U, x, V), the WLRA R-convex fiber is
    T_U V_k(R^m) x B_r(-x) x T_V V_k(R^n), so only the middle Euclidean
    block is modified.
    """
    if r < 0:
        raise ValueError("Radius r must be non-negative.")
    _, x, _ = validate_point(point)
    xi_U, y, xi_V = validate_tangent(tangent, point)
    # Leave the two Stiefel tangent blocks unchanged and project the middle block
    # onto the admissible Euclidean ball.
    return xi_U, _project_to_ball(y, -x, r), xi_V
