from __future__ import annotations

import torch

from manifold_opt.geometry.types import Point, Tangent, validate_tangent


def inner_product(point: Point, u: Tangent, v: Tangent) -> torch.Tensor:
    """
    Compute the product Riemannian inner product at point = (U, x, V).

    The metric is the blockwise Frobenius/Euclidean product:
    <u, v>_point = trace(u_U^T v_U) + u_x^T v_x + trace(u_V^T v_V).
    """
    u_U, u_x, u_V = validate_tangent(u, point, name="u")
    v_U, v_x, v_V = validate_tangent(v, point, name="v")

    # Use the induced product-manifold metric for the gradient-direction pairing.
    metric = torch.sum(u_U * v_U) + torch.sum(u_x * v_x) + torch.sum(u_V * v_V)
    if not torch.isfinite(metric):
        raise ValueError("inner_product produced a non-finite value.")
    return metric
