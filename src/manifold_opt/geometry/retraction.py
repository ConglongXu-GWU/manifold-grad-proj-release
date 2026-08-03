from __future__ import annotations

import torch

from manifold_opt.geometry.types import Point, Tangent, validate_point, validate_tangent


def qr(x: torch.Tensor) -> torch.Tensor:
    """
    Compute QR decomposition and return Q with sign-fix so diag(R) >= 0.

    Requires x to have shape (m, k) with m >= k.
    """
    if x.ndim != 2:
        raise ValueError(f"qr(): expected a 2D tensor, got shape {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("qr(): input must contain only finite values.")

    m, k = x.shape
    if m < k:
        raise ValueError(
            f"qr(): invalid shape {m} x {k}. "
            f"Cannot produce {k} orthonormal columns in R^{m}. "
            f"Require m >= k."
        )

    q, r = torch.linalg.qr(x, mode="reduced")
    d = torch.diagonal(r)
    sign = torch.where(d < 0, -1.0, 1.0)
    q = q * sign.unsqueeze(0)
    if not torch.isfinite(q).all():
        raise ValueError("qr() produced non-finite values.")
    return q


def retraction_qr(point: Point, step: Tangent) -> Point:
    """
    Retract point = (U, x, V) along step = (xi_U, x_hat, xi_V).

    QR retraction is used for the two column-Stiefel factors. The middle
    Euclidean block is updated by addition.
    """
    U, x, V = validate_point(point)
    xi_U, x_hat, xi_V = validate_tangent(step, (U, x, V), name="step")
    # Apply the QR product retraction blockwise, with an additive middle block.
    point_next = (qr(U + xi_U), x + x_hat, qr(V + xi_V))
    if not all(torch.isfinite(block).all() for block in point_next):
        raise ValueError("retraction produced non-finite values.")
    return point_next


def polar(x: torch.Tensor, xi: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the Stiefel polar retraction R_x(xi).

    For a point x on the column-Stiefel manifold and tangent step xi, the
    polar retraction is

        R_x(xi) = (x + xi) ((x + xi)^T (x + xi))^{-1/2}.

    Since xi is tangent, x^T xi + xi^T x = 0, so
    (x + xi)^T (x + xi) = I + xi^T xi. This implementation uses that
    symmetric positive definite k x k factor and computes its inverse square
    root by eigendecomposition.

    Requires x to have shape (m, k) with m >= k.
    """
    if x.ndim != 2:
        raise ValueError(f"polar(): expected a 2D tensor, got shape {tuple(x.shape)}")
    if xi.shape != x.shape:
        raise ValueError(f"polar(): x and xi must have same shape, got {tuple(x.shape)} vs {tuple(xi.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("polar(): input must contain only finite values.")
    if not torch.isfinite(xi).all():
        raise ValueError("polar(): xi must contain only finite values.")

    m, k = x.shape
    if m < k:
        raise ValueError(
            f"polar(): invalid shape {m} x {k}. "
            f"Cannot produce {k} orthonormal columns in R^{m}. "
            f"Require m >= k."
        )
    eye_n = torch.eye(k, device=x.device, dtype=x.dtype)
    a = x + xi
    b = eye_n + xi.T @ xi

    if b.device.type == "mps":
        eigvals, eigvecs = torch.linalg.eigh(b.cpu())
        eigvals = eigvals.to(device=b.device)
        eigvecs = eigvecs.to(device=b.device)
    else:
        eigvals, eigvecs = torch.linalg.eigh(b)
    eigvals = torch.clamp(eigvals, min=eps)
    aq = a @ eigvecs
    aq_scaled = aq * eigvals.rsqrt().unsqueeze(-2)
    transformed = aq_scaled @ eigvecs.transpose(-1, -2)

    if not torch.isfinite(transformed).all():
        raise ValueError("polar() produced non-finite values.")
    return transformed


def retraction_polar(point: Point, step: Tangent) -> Point:
    """
    Retract point = (U, x, V) along step = (xi_U, x_hat, xi_V).

    Polar retraction is used for the two column-Stiefel factors. The middle
    Euclidean block is updated by addition.
    """
    U, x, V = validate_point(point)
    xi_U, x_hat, xi_V = validate_tangent(step, (U, x, V), name="step")
    # Apply the polar product retraction blockwise, with an additive middle block.
    point_next = (polar(U, xi_U), x + x_hat, polar(V, xi_V))
    if not all(torch.isfinite(block).all() for block in point_next):
        raise ValueError("retraction produced non-finite values.")
    return point_next


retraction = retraction_qr
