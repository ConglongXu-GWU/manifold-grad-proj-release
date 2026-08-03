
from typing import Callable, Dict, List, Tuple
import torch
import copy

from manifold_opt.line_search.armijo import armijo_feasible_direction, armijo_projection_arc, armijo_wlra_reg
from manifold_opt.optim.initial import initial_lr


RMSEMode = str


def _validate_retraction(name: str) -> None:
    """Validate a retraction supported by the Armijo optimization loops."""
    if name not in {"qr", "polar"}:
        raise ValueError("retraction must be one of 'qr' or 'polar'.")


def _validate_rmse_mode(rmse_mode: RMSEMode) -> None:
    if rmse_mode not in {"history", "final"}:
        raise ValueError("rmse_mode must be 'history' or 'final'.")


def _missing_entries(a_full: torch.Tensor, w: torch.Tensor, *, dtype: torch.dtype, device: torch.device):
    miss = (1.0 - w).to(dtype=dtype, device=device)
    den = miss.sum().clamp_min(1.0)
    return miss, den


def _rmse_on_missing(point, a_full: torch.Tensor, miss: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    U, x, V = point
    err = miss * (a_full - (U * x.unsqueeze(0)) @ V.transpose(0, 1))
    return (err.pow(2).sum() / den).sqrt()


# the Armijo gradient-descent loop for regularized WLRA
def optimize_reg_armijo(
    a_full: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    k: int,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    initial_fn: Callable[
        [torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = initial_lr,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    retraction: str = "qr",
    rmse_mode: RMSEMode = "history",
    diagnostics: Dict[str, List[int]] | None = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """
    Armijo gradient-descent optimization for regularized WLRA (PyTorch).

    The step is selected by armijo_wlra_reg(...). When diagnostics is provided,
    it records per-iteration Armijo inequality evaluation counts.
    """

    # --- sanity checks ---
    if a.shape != w.shape or a.shape != a_full.shape:
        raise ValueError(
            f"optimize_reg_armijo(): a, w, a_full must have the same shape. "
            f"Got a={tuple(a.shape)}, w={tuple(w.shape)}, a_full={tuple(a_full.shape)}"
        )
    if a.ndim != 2:
        raise ValueError(f"optimize_reg_armijo(): expected 2D tensors, got a.ndim={a.ndim}")
    _validate_rmse_mode(rmse_mode)
    _validate_retraction(retraction)

    losses: List[torch.Tensor] = []
    rmses: List[torch.Tensor] = []
    miss, den = _missing_entries(a_full, w, dtype=a.dtype, device=a.device)

    # --- initialization ---
    U, x, V = initial_fn(a, k)
    U = copy.deepcopy(U).to(device=a.device, dtype=a.dtype)
    x = copy.deepcopy(x).to(device=a.device, dtype=a.dtype)
    V = copy.deepcopy(V).to(device=a.device, dtype=a.dtype)
    point = (U, x, V)
    gradient = tuple(torch.zeros_like(block) for block in point)

    # --- optimization loop ---
    for _ in range(iteration_numbers):
        armijo_result = armijo_wlra_reg(
            point,
            a,
            w,
            lmbda=lmbda,
            s=lr,
            beta=beta,
            sigma=sigma,
            max_backtracks=max_backtracks,
            armijo_tol=armijo_tol,
            retraction=retraction,
        )
        if diagnostics is not None:
            diagnostics.setdefault("armijo_evaluation_history", []).append(armijo_result.armijo_evaluations)
        if not armijo_result.accepted:
            raise RuntimeError(f"optimize_reg_armijo(): Armijo step failed: {armijo_result.reason}")

        gradient = armijo_result.grad
        loss = armijo_result.f_current
        point = armijo_result.point_next
        U, x, V = point

        if rmse_mode == "history":
            rmses.append(_rmse_on_missing(point, a_full, miss, den))

        losses.append(loss)

    if rmse_mode == "final":
        rmses.append(_rmse_on_missing(point, a_full, miss, den))

    params: Dict[str, torch.Tensor] = {"U": U, "x": x, "V": V}
    grads_out: Dict[str, torch.Tensor] = {"xi_U": gradient[0], "x_hat": gradient[1], "xi_V": gradient[2]}

    return losses, rmses, params, grads_out



# gradient-descent loop for WLRA with R-convex constraint and Armijo stepsize
def optimize_constraint_armijo(
    a_full: torch.Tensor,
    a: torch.Tensor,
    w: torch.Tensor,
    k: int,
    r: float = 100,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    initial_fn: Callable[
        [torch.Tensor, int],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = initial_lr,
    armijo_rule: str = "feasible_direction",
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    retraction: str = "qr",
    rmse_mode: RMSEMode = "history",
    diagnostics: Dict[str, List[int]] | None = None,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """
    Projected-gradient optimization for WLRA with a ball constraint on x.

    Model:
        p = U diag(x) V^T

    Objective (data-fit only):
        L = sum_{i,j} w_ij (a_ij - p_ij)^2

    Constraints:
        U has orthonormal columns (U^T U = I)
        V has orthonormal columns (V^T V = I)
        The middle tangent block is projected onto the Euclidean ball
        centered at -x with radius r.

    RMSE:
        Computed ONLY on missed entries (where w == 0).

    Diagnostics:
        If diagnostics is provided, the loop appends each iteration's actual
        Armijo inequality evaluation count to diagnostics["armijo_evaluation_history"].
    """

    # --- sanity checks ---
    if a.shape != w.shape or a.shape != a_full.shape:
        raise ValueError(
            f"optimize_constraint_armijo(): a, w, a_full must have the same shape. "
            f"Got a={tuple(a.shape)}, w={tuple(w.shape)}, a_full={tuple(a_full.shape)}"
        )
    if a.ndim != 2:
        raise ValueError(f"optimize_constraint_armijo(): expected 2D tensors, got a.ndim={a.ndim}")
    if r < 0:
        raise ValueError("optimize_constraint_armijo(): r must be non-negative.")
    if armijo_rule not in {"feasible_direction", "projection_arc"}:
        raise ValueError("optimize_constraint_armijo(): armijo_rule must be 'feasible_direction' or 'projection_arc'.")
    _validate_rmse_mode(rmse_mode)
    _validate_retraction(retraction)

    losses: List[torch.Tensor] = []
    rmses: List[torch.Tensor] = []
    miss, den = _missing_entries(a_full, w, dtype=a.dtype, device=a.device)

    # --- initialization ---
    U, x, V = initial_fn(a, k)
    U = copy.deepcopy(U).to(device=a.device, dtype=a.dtype)
    x = copy.deepcopy(x).to(device=a.device, dtype=a.dtype)
    V = copy.deepcopy(V).to(device=a.device, dtype=a.dtype)
    point = (U, x, V)
    gradient = tuple(torch.zeros_like(block) for block in point)

    # --- optimization loop ---
    for _ in range(iteration_numbers):
        if armijo_rule == "feasible_direction":
            # Keep s fixed and backtrack only on alpha = beta^m.
            armijo_result = armijo_feasible_direction(
                point,
                a,
                w,
                r=r,
                s=lr,
                beta=beta,
                sigma=sigma,
                max_backtracks=max_backtracks,
                armijo_tol=armijo_tol,
                retraction=retraction,
            )
        else:
            # Backtrack on s = beta^m * s_bar while keeping alpha = 1.
            armijo_result = armijo_projection_arc(
                point,
                a,
                w,
                r=r,
                s_bar=lr,
                beta=beta,
                sigma=sigma,
                max_backtracks=max_backtracks,
                armijo_tol=armijo_tol,
                retraction=retraction,
            )
        if diagnostics is not None:
            diagnostics.setdefault("armijo_evaluation_history", []).append(armijo_result.armijo_evaluations)
        if not armijo_result.accepted:
            raise RuntimeError(f"optimize_constraint_armijo(): Armijo step failed: {armijo_result.reason}")

        gradient = armijo_result.grad
        loss = armijo_result.f_current
        # Assign the accepted retracted point to the next iterate.
        point = armijo_result.point_next
        U, x, V = point

        if rmse_mode == "history":
            rmses.append(_rmse_on_missing(point, a_full, miss, den))

        losses.append(loss)

    if rmse_mode == "final":
        rmses.append(_rmse_on_missing(point, a_full, miss, den))

    params: Dict[str, torch.Tensor] = {"U": U, "x": x, "V": V}
    grads_out: Dict[str, torch.Tensor] = {"xi_U": gradient[0], "x_hat": gradient[1], "xi_V": gradient[2]}
    return losses, rmses, params, grads_out

