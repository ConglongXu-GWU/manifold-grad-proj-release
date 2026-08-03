from __future__ import annotations

import torch

from manifold_opt.geometry.types import Point, Tangent, validate_point, validate_wlra_data


def _orthogonal_proj(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Orthogonally project z onto the column-Stiefel tangent space at x.

    Assumes x has orthonormal columns. Shape and finite-value checks are
    enforced, but orthonormality itself is left to callers.
    """
    if x.shape != z.shape:
        raise ValueError(f"_orthogonal_proj(): x and z must have same shape, got {tuple(x.shape)} vs {tuple(z.shape)}")

    m = x.transpose(0, 1) @ z
    sym = 0.5 * (m + m.transpose(0, 1))
    # Orthogonally project the ambient block onto the Stiefel tangent space.
    projected = z - x @ sym
    if not torch.isfinite(projected).all():
        raise ValueError("_orthogonal_proj() produced non-finite values.")
    return projected


def grad(point: Point, a: torch.Tensor, w: torch.Tensor) -> Tangent:
    """
    Compute the Riemannian gradient of WLRA at point = (U, x, V).

    The point convention is U in R^{m x k}, x in R^k, and V in R^{n x k};
    the model is p = U diag(x) V^T. The returned tangent is
    (xi_U, x_hat, xi_V).
    """
    U, x, V = validate_point(point)
    validate_wlra_data((U, x, V), a, w)

    p = (U * x.unsqueeze(0)) @ V.transpose(0, 1)
    dp = -2.0 * w * (a - p)

    gradU = dp @ (V * x.unsqueeze(0))
    gradV = dp.transpose(0, 1) @ (U * x.unsqueeze(0))
    mid = U.transpose(0, 1) @ dp @ V
    gradx = torch.diagonal(mid)

    tangent = (_orthogonal_proj(U, gradU), gradx, _orthogonal_proj(V, gradV))
    if not all(torch.isfinite(block).all() for block in tangent):
        raise ValueError("grad produced non-finite values.")
    return tangent


def grad_reg(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    lmbda: float | torch.Tensor,
) -> Tangent:
    """
    Compute the Riemannian gradient of regularized WLRA at point = (U, x, V).

    The regularizer is lmbda * ||x||_2^2, so only the middle tangent block
    differs from grad(...).
    """
    U, x, V = validate_point(point)
    validate_wlra_data((U, x, V), a, w)
    if not torch.is_tensor(lmbda):
        lmbda_t = torch.tensor(lmbda, device=a.device, dtype=a.dtype)
    else:
        lmbda_t = lmbda.to(device=a.device, dtype=a.dtype)
    if not torch.isfinite(lmbda_t).all():
        raise ValueError("lmbda must contain only finite values.")

    xi_U, x_hat, xi_V = grad((U, x, V), a, w)
    tangent = (xi_U, x_hat + 2.0 * lmbda_t * x, xi_V)
    if not all(torch.isfinite(block).all() for block in tangent):
        raise ValueError("grad_reg produced non-finite values.")
    return tangent
    
    
