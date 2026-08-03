from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.projection import feasible_set_projection
from manifold_opt.geometry.retraction import retraction_polar, retraction_qr
from manifold_opt.geometry.types import Point, Tangent
from manifold_opt.objectives.first_order import wlra_first_order, wlra_reg_first_order
from manifold_opt.objectives.losses import wlra_loss, wlra_loss_reg


Scalar = float | torch.Tensor
RetractionName = str
RetractionFn = Callable[[Point, Tangent], Point]


@dataclass(frozen=True)
class ArmijoResult:
    accepted: bool
    point_next: Point
    step: Tangent
    m: int
    alpha: float
    s: float
    f_current: torch.Tensor
    f_next: torch.Tensor
    armijo_lhs: torch.Tensor
    armijo_rhs: torch.Tensor
    descent_inner: torch.Tensor
    grad: Tangent
    reason: str
    armijo_evaluations: int = 0


def armijo_lhs(f_current: Scalar, f_trial: Scalar) -> Scalar:
    """Compute the Armijo left-hand side f_current - f_trial."""
    # Difference between the current and trial objective values.
    return f_current - f_trial


def armijo_rhs_feasible_direction(
    point: Point,
    grad: Tangent,
    projected_direction: Tangent,
    alpha: float,
    sigma: float,
) -> torch.Tensor:
    """Compute -sigma * alpha * <grad, projected_direction>_point."""
    # Sufficient-decrease threshold for a scaled projected direction.
    return -sigma * alpha * inner_product(point, grad, projected_direction)


def armijo_rhs_projection_arc(
    point: Point,
    grad: Tangent,
    projected_step: Tangent,
    sigma: float,
) -> torch.Tensor:
    """Compute -sigma * <grad, projected_step>_point."""
    # Sufficient-decrease threshold for the current projected step.
    return -sigma * inner_product(point, grad, projected_step)


def satisfies_armijo(lhs: Scalar, rhs: Scalar, tol: float = 1e-12) -> bool:
    """Return whether lhs + tol >= rhs."""
    return bool(lhs + tol >= rhs)


def make_trial_step_feasible_direction(
    point: Point,
    grad: Tangent,
    s: float,
    alpha: float,
    r: float,
) -> tuple[Tangent, Tangent]:
    """
    Construct the fixed projected direction and scaled feasible-direction step.

    The repository projection requires the WLRA ball radius r. Backtracking
    callers should compute projected_direction once for a fixed s, then rescale
    the returned direction by alpha.
    """
    # Project the fixed descent vector once.
    descent = tuple(-s * block for block in grad)
    projected_direction = feasible_set_projection(point, descent, r)
    # Scale the direction after projection.
    step = tuple(alpha * block for block in projected_direction)
    return projected_direction, step


def make_trial_step_projection_arc(
    point: Point,
    grad: Tangent,
    s: float,
    r: float,
) -> Tangent:
    """Construct Pi_{C^r_point}(-s * grad) for the current projection-arc s."""
    # Recompute the feasible projection for the current candidate step size.
    descent = tuple(-s * block for block in grad)
    return feasible_set_projection(point, descent, r)


def evaluate_trial_point(
    point: Point,
    step: Tangent,
    a: torch.Tensor,
    w: torch.Tensor,
    retract: RetractionFn | None = None,
) -> tuple[Point, torch.Tensor]:
    """Retract point along step, then evaluate wlra_loss at the trial point."""
    if retract is None:
        retract = retraction_qr
    # Retract the candidate tangent step.
    point_trial = retract(point, step)
    # Evaluate the objective at the retracted point.
    f_trial = wlra_loss(point_trial, a, w)
    return point_trial, f_trial


def is_zero_step(step: Tangent, tol: float = 1e-12) -> bool:
    """Return whether the tangent step has blockwise Euclidean norm <= tol."""
    if not isinstance(step, tuple) or len(step) != 3:
        raise ValueError("step must be a tuple (xi_U, x_hat, xi_V).")
    if not all(torch.is_tensor(block) for block in step):
        raise ValueError("step must contain only torch.Tensor objects.")
    if not all(torch.isfinite(block).all() for block in step):
        raise ValueError("step must contain only finite values.")

    norm_sq = sum(torch.sum(block * block) for block in step)
    return bool(torch.sqrt(norm_sq) <= tol)


def armijo_residual(lhs: Scalar, rhs: Scalar) -> Scalar:
    """Compute the Armijo residual lhs - rhs."""
    return lhs - rhs


def _validate_common_parameters(beta: float, sigma: float, max_backtracks: int) -> None:
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1).")
    if not 0.0 < sigma < 1.0:
        raise ValueError("sigma must be in (0, 1).")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative.")


def _zero_step_like(point: Point) -> Tangent:
    return tuple(torch.zeros_like(block) for block in point)


def resolve_retraction(retraction: RetractionName = "qr") -> RetractionFn:
    """Return a tangent-step WLRA retraction selected by name."""
    if retraction == "qr":
        return retraction_qr
    if retraction == "polar":
        return retraction_polar
    raise ValueError("retraction must be one of 'qr' or 'polar'.")


def _validate_retraction_name(retraction: RetractionName) -> None:
    if retraction not in {"qr", "polar"}:
        raise ValueError("retraction must be one of 'qr' or 'polar'.")


def armijo_feasible_direction(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    r: float,
    s: float,
    beta: float,
    sigma: float,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    zero_tol: float = 1e-12,
    retraction: RetractionName = "qr",
) -> ArmijoResult:
    """
    Armijo rule along the feasible direction for WLRA.

    Computes d = Pi_{C^r_point}(-s * grad) once for fixed s, then backtracks
    only on alpha = beta**m.
    """
    _validate_common_parameters(beta, sigma, max_backtracks)
    if s <= 0:
        raise ValueError("s must be positive.")
    _validate_retraction_name(retraction)
    retract = resolve_retraction(retraction)

    # Evaluate the current objective and gradient once.
    evaluation = wlra_first_order(point, a, w)
    gradient = evaluation.riemannian_gradient
    f_current = evaluation.objective
    # Compute the projected direction once for the fixed step size.
    projected_direction, _ = make_trial_step_feasible_direction(point, gradient, s, 1.0, r)
    descent_inner = inner_product(point, gradient, projected_direction)

    if is_zero_step(projected_direction, tol=zero_tol):
        zero_step = _zero_step_like(point)
        zero = f_current - f_current
        return ArmijoResult(
            accepted=True,
            point_next=point,
            step=zero_step,
            m=0,
            alpha=1.0,
            s=s,
            f_current=f_current,
            f_next=f_current,
            armijo_lhs=zero,
            armijo_rhs=-sigma * descent_inner,
            descent_inner=descent_inner,
            grad=gradient,
            reason="stationary_zero_step",
            armijo_evaluations=0,
        )

    last_result = None
    armijo_evaluations = 0
    # Test candidates in increasing m so the first accepted index is minimal.
    for m in range(max_backtracks + 1):
        alpha = beta**m
        # Scale the already projected direction by beta**m.
        step = tuple(alpha * block for block in projected_direction)
        point_trial, f_trial = evaluate_trial_point(point, step, a, w, retract)
        # Compare the two sides of the sufficient-decrease condition.
        lhs = armijo_lhs(f_current, f_trial)
        rhs = armijo_rhs_feasible_direction(point, gradient, projected_direction, alpha, sigma)
        armijo_evaluations += 1
        accepted = satisfies_armijo(lhs, rhs, armijo_tol)
        result = ArmijoResult(
            accepted=accepted,
            point_next=point_trial,
            step=step,
            m=m,
            alpha=alpha,
            s=s,
            f_current=f_current,
            f_next=f_trial,
            armijo_lhs=lhs,
            armijo_rhs=rhs,
            descent_inner=descent_inner,
            grad=gradient,
            reason="accepted" if accepted else "armijo_failed",
            armijo_evaluations=armijo_evaluations,
        )
        if result.accepted:
            # Return the first admissible retracted point.
            return result
        last_result = result

    return ArmijoResult(
        accepted=False,
        point_next=last_result.point_next,
        step=last_result.step,
        m=last_result.m,
        alpha=last_result.alpha,
        s=s,
        f_current=f_current,
        f_next=last_result.f_next,
        armijo_lhs=last_result.armijo_lhs,
        armijo_rhs=last_result.armijo_rhs,
        descent_inner=descent_inner,
        grad=gradient,
        reason="max_backtracks_exceeded",
        armijo_evaluations=last_result.armijo_evaluations,
    )


def armijo_projection_arc(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    r: float,
    s_bar: float,
    beta: float,
    sigma: float,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    zero_tol: float = 1e-12,
    retraction: RetractionName = "qr",
) -> ArmijoResult:
    """
    Armijo rule along the projection arc for WLRA.

    Recomputes Pi_{C^r_point}(-s * grad) for each candidate
    s = beta**m * s_bar and always uses alpha = 1.
    """
    _validate_common_parameters(beta, sigma, max_backtracks)
    if s_bar <= 0:
        raise ValueError("s_bar must be positive.")
    _validate_retraction_name(retraction)
    retract = resolve_retraction(retraction)

    # Evaluate the current objective and gradient once.
    evaluation = wlra_first_order(point, a, w)
    gradient = evaluation.riemannian_gradient
    f_current = evaluation.objective
    last_result = None
    armijo_evaluations = 0

    # Test candidates in increasing m so the first accepted index is minimal.
    for m in range(max_backtracks + 1):
        # Backtrack the projection parameter as beta**m * s_bar.
        candidate_s = beta**m * s_bar
        # Recompute the feasible projection for every candidate step size.
        step = make_trial_step_projection_arc(point, gradient, candidate_s, r)
        descent_inner = inner_product(point, gradient, step)

        if is_zero_step(step, tol=zero_tol):
            zero_step = _zero_step_like(point)
            zero = f_current - f_current
            return ArmijoResult(
                accepted=True,
                point_next=point,
                step=zero_step,
                m=m,
                alpha=1.0,
                s=candidate_s,
                f_current=f_current,
                f_next=f_current,
                armijo_lhs=zero,
                armijo_rhs=-sigma * descent_inner,
                descent_inner=descent_inner,
                grad=gradient,
                reason="stationary_zero_step",
                armijo_evaluations=armijo_evaluations,
            )

        point_trial, f_trial = evaluate_trial_point(point, step, a, w, retract)
        # Compare the two sides of the sufficient-decrease condition.
        lhs = armijo_lhs(f_current, f_trial)
        rhs = armijo_rhs_projection_arc(point, gradient, step, sigma)
        armijo_evaluations += 1
        accepted = satisfies_armijo(lhs, rhs, armijo_tol)
        result = ArmijoResult(
            accepted=accepted,
            point_next=point_trial,
            step=step,
            m=m,
            alpha=1.0,
            s=candidate_s,
            f_current=f_current,
            f_next=f_trial,
            armijo_lhs=lhs,
            armijo_rhs=rhs,
            descent_inner=descent_inner,
            grad=gradient,
            reason="accepted" if accepted else "armijo_failed",
            armijo_evaluations=armijo_evaluations,
        )
        if result.accepted:
            # Return the first admissible retracted point.
            return result
        last_result = result

    return ArmijoResult(
        accepted=False,
        point_next=last_result.point_next,
        step=last_result.step,
        m=last_result.m,
        alpha=1.0,
        s=last_result.s,
        f_current=f_current,
        f_next=last_result.f_next,
        armijo_lhs=last_result.armijo_lhs,
        armijo_rhs=last_result.armijo_rhs,
        descent_inner=last_result.descent_inner,
        grad=gradient,
        reason="max_backtracks_exceeded",
        armijo_evaluations=last_result.armijo_evaluations,
    )


def armijo_wlra_reg(
    point: Point,
    a: torch.Tensor,
    w: torch.Tensor,
    lmbda: float | torch.Tensor,
    s: float,
    beta: float,
    sigma: float,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    zero_tol: float = 1e-12,
    retraction: RetractionName = "qr",
) -> ArmijoResult:
    """
    Armijo rule for regularized WLRA without feasible projection.

    Computes d = -s * grad_reg(point) once for fixed s, then backtracks only
    on alpha = beta**m before retracting the scaled tangent step.
    """
    _validate_common_parameters(beta, sigma, max_backtracks)
    if s <= 0:
        raise ValueError("s must be positive.")
    _validate_retraction_name(retraction)
    retract = resolve_retraction(retraction)

    evaluation = wlra_reg_first_order(point, a, w, lmbda)
    gradient = evaluation.riemannian_gradient
    f_current = evaluation.objective
    direction = tuple(-s * block for block in gradient)
    descent_inner = inner_product(point, gradient, direction)

    if is_zero_step(direction, tol=zero_tol):
        zero_step = _zero_step_like(point)
        zero = f_current - f_current
        return ArmijoResult(
            accepted=True,
            point_next=point,
            step=zero_step,
            m=0,
            alpha=1.0,
            s=s,
            f_current=f_current,
            f_next=f_current,
            armijo_lhs=zero,
            armijo_rhs=-sigma * descent_inner,
            descent_inner=descent_inner,
            grad=gradient,
            reason="stationary_zero_step",
            armijo_evaluations=0,
        )

    last_result = None
    armijo_evaluations = 0
    for m in range(max_backtracks + 1):
        alpha = beta**m
        step = tuple(alpha * block for block in direction)
        point_trial = retract(point, step)
        f_trial = wlra_loss_reg(point_trial, a, w, lmbda)
        lhs = armijo_lhs(f_current, f_trial)
        rhs = -sigma * alpha * descent_inner
        armijo_evaluations += 1
        accepted = satisfies_armijo(lhs, rhs, armijo_tol)
        result = ArmijoResult(
            accepted=accepted,
            point_next=point_trial,
            step=step,
            m=m,
            alpha=alpha,
            s=s,
            f_current=f_current,
            f_next=f_trial,
            armijo_lhs=lhs,
            armijo_rhs=rhs,
            descent_inner=descent_inner,
            grad=gradient,
            reason="accepted" if accepted else "armijo_failed",
            armijo_evaluations=armijo_evaluations,
        )
        if result.accepted:
            return result
        last_result = result

    return ArmijoResult(
        accepted=False,
        point_next=last_result.point_next,
        step=last_result.step,
        m=last_result.m,
        alpha=last_result.alpha,
        s=s,
        f_current=f_current,
        f_next=last_result.f_next,
        armijo_lhs=last_result.armijo_lhs,
        armijo_rhs=last_result.armijo_rhs,
        descent_inner=descent_inner,
        grad=gradient,
        reason="max_backtracks_exceeded",
        armijo_evaluations=last_result.armijo_evaluations,
    )
