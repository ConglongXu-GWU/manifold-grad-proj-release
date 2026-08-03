
from __future__ import annotations

import torch

from manifold_opt.geometry.types import Point, validate_point, validate_wlra_data


def _reconstruction(point: Point) -> torch.Tensor:
    U, x, V = validate_point(point)
    return (U * x.unsqueeze(0)) @ V.transpose(0, 1)


def wlra_loss(point: Point, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Weighted low-rank approximation loss for point = (U, x, V).

    The point convention is U in R^{m x k}, x in R^k, and V in R^{n x k};
    the model is p = U diag(x) V^T.
    """
    validate_wlra_data(point, a, w)
    # Reconstruct P = U diag(x) V^T and evaluate the weighted data-fit objective.
    p = _reconstruction(point)
    loss = (w * (a - p).pow(2)).sum()
    if not torch.isfinite(loss):
        raise ValueError("wlra_loss produced a non-finite value.")
    return loss


def wlra_loss_reg(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    lmbda: float | torch.Tensor,
) -> torch.Tensor:
    """
    Regularized WLRA loss for point = (U, x, V).

    The regularizer is lmbda * ||x||_2^2.
    """
    U, x, V = validate_point(point)
    validate_wlra_data((U, x, V), a, w)
    p = _reconstruction((U, x, V))
    data_fit = (w * (a - p).pow(2)).sum()

    if not torch.is_tensor(lmbda):
        lmbda = torch.tensor(lmbda, device=p.device, dtype=p.dtype)
    else:
        lmbda = lmbda.to(device=p.device, dtype=p.dtype)
    if not torch.isfinite(lmbda).all():
        raise ValueError("lmbda must contain only finite values.")

    loss = data_fit + lmbda * x.pow(2).sum()
    if not torch.isfinite(loss):
        raise ValueError("wlra_loss_reg produced a non-finite value.")
    return loss
