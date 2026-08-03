from __future__ import annotations

from dataclasses import dataclass

import torch

from manifold_opt.geometry.riemannian_gradient import _orthogonal_proj
from manifold_opt.geometry.types import Point, Tangent, validate_point, validate_wlra_data


@dataclass(frozen=True)
class WlraFirstOrder:
    data_loss: torch.Tensor
    objective: torch.Tensor
    riemannian_gradient: Tangent


def _as_lambda_tensor(lmbda: float | torch.Tensor, *, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(lmbda):
        result = torch.tensor(lmbda, device=x.device, dtype=x.dtype)
    else:
        result = lmbda.to(device=x.device, dtype=x.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("lmbda must contain only finite values.")
    return result


def wlra_first_order(point: Point, a: torch.Tensor, w: torch.Tensor) -> WlraFirstOrder:
    """Evaluate WLRA loss and first-order data with one reconstruction."""
    U, x, V = validate_point(point)
    validate_wlra_data((U, x, V), a, w)

    # Reconstruct P = U diag(x) V^T and evaluate the weighted data-fit objective.
    reconstruction = (U * x.unsqueeze(0)) @ V.transpose(0, 1)
    residual = a - reconstruction
    weighted_residual = w * residual
    data_loss = (weighted_residual * residual).sum()
    if not torch.isfinite(data_loss):
        raise ValueError("wlra loss produced a non-finite value.")

    dp = -2.0 * weighted_residual
    grad_u = dp @ (V * x.unsqueeze(0))
    grad_v = dp.transpose(0, 1) @ (U * x.unsqueeze(0))
    mid = U.transpose(0, 1) @ dp @ V
    grad_x = torch.diagonal(mid)

    # Project the ambient Stiefel blocks to obtain the Riemannian gradient.
    riemannian_gradient = (_orthogonal_proj(U, grad_u), grad_x, _orthogonal_proj(V, grad_v))
    if not all(torch.isfinite(block).all() for block in riemannian_gradient):
        raise ValueError("WLRA first-order evaluation produced non-finite values.")
    return WlraFirstOrder(
        data_loss=data_loss,
        objective=data_loss,
        riemannian_gradient=riemannian_gradient,
    )


def wlra_reg_first_order(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    lmbda: float | torch.Tensor,
) -> WlraFirstOrder:
    """Evaluate regularized WLRA loss and first-order data with one reconstruction."""
    U, x, V = validate_point(point)
    evaluation = wlra_first_order((U, x, V), a, w)
    lmbda_t = _as_lambda_tensor(lmbda, x=x)

    xi_u, x_hat, xi_v = evaluation.riemannian_gradient
    objective = evaluation.data_loss + lmbda_t * x.pow(2).sum()
    if not torch.isfinite(objective):
        raise ValueError("regularized WLRA loss produced a non-finite value.")

    riemannian_gradient = (xi_u, x_hat + 2.0 * lmbda_t * x, xi_v)
    if not all(torch.isfinite(block).all() for block in riemannian_gradient):
        raise ValueError("regularized WLRA first-order evaluation produced non-finite values.")
    return WlraFirstOrder(
        data_loss=evaluation.data_loss,
        objective=objective,
        riemannian_gradient=riemannian_gradient,
    )
