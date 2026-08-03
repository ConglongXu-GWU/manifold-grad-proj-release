from typing import Tuple

import torch

Point = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Tangent = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def validate_point(point: Point, *, name: str = "point") -> Point:
    """
    Validate the WLRA manifold point convention.

    A point is represented as (U, x, V), where U has shape (m, k), x has
    shape (k,), and V has shape (n, k). Orthogonality is assumed by callers
    and is not checked here.
    """
    if not isinstance(point, tuple) or len(point) != 3:
        raise ValueError(f"{name} must be a tuple (U, x, V).")

    U, x, V = point
    if not all(torch.is_tensor(tensor) for tensor in point):
        raise ValueError(f"{name} must contain only torch.Tensor objects.")
    if U.ndim != 2:
        raise ValueError(f"{name}[0] must be a 2D tensor, got shape {tuple(U.shape)}.")
    if x.ndim != 1:
        raise ValueError(f"{name}[1] must be a 1D tensor, got shape {tuple(x.shape)}.")
    if V.ndim != 2:
        raise ValueError(f"{name}[2] must be a 2D tensor, got shape {tuple(V.shape)}.")

    k = x.shape[0]
    if U.shape[1] != k or V.shape[1] != k:
        raise ValueError(
            f"{name} has inconsistent rank dimensions: "
            f"U={tuple(U.shape)}, x={tuple(x.shape)}, V={tuple(V.shape)}."
        )

    _require_finite(f"{name}[0]", U)
    _require_finite(f"{name}[1]", x)
    _require_finite(f"{name}[2]", V)
    return U, x, V


def validate_tangent(tangent: Tangent, point: Point, *, name: str = "tangent") -> Tangent:
    """
    Validate the WLRA tangent convention against a point.

    A tangent is represented as (xi_U, x_hat, xi_V) with block shapes exactly
    matching (U, x, V). Tangency to the Stiefel factors is assumed by callers
    and is not checked here.
    """
    U, x, V = validate_point(point)
    if not isinstance(tangent, tuple) or len(tangent) != 3:
        raise ValueError(f"{name} must be a tuple (xi_U, x_hat, xi_V).")

    xi_U, x_hat, xi_V = tangent
    if not all(torch.is_tensor(tensor) for tensor in tangent):
        raise ValueError(f"{name} must contain only torch.Tensor objects.")
    if xi_U.shape != U.shape or x_hat.shape != x.shape or xi_V.shape != V.shape:
        raise ValueError(
            f"{name} shapes must match point shapes: "
            f"tangent={(tuple(xi_U.shape), tuple(x_hat.shape), tuple(xi_V.shape))}, "
            f"point={(tuple(U.shape), tuple(x.shape), tuple(V.shape))}."
        )

    _require_finite(f"{name}[0]", xi_U)
    _require_finite(f"{name}[1]", x_hat)
    _require_finite(f"{name}[2]", xi_V)
    return xi_U, x_hat, xi_V


def validate_wlra_data(point: Point, a: torch.Tensor, w: torch.Tensor) -> None:
    U, _, V = validate_point(point)
    if a.shape != w.shape:
        raise ValueError(f"a and w must have the same shape, got {tuple(a.shape)} vs {tuple(w.shape)}.")
    if a.ndim != 2:
        raise ValueError(f"a and w must be 2D tensors, got a.ndim={a.ndim}.")
    if a.shape != (U.shape[0], V.shape[0]):
        raise ValueError(
            f"a and w must have shape (m, n) matching point blocks, "
            f"got a={tuple(a.shape)}, U={tuple(U.shape)}, V={tuple(V.shape)}."
        )
    _require_finite("a", a)
    _require_finite("w", w)
