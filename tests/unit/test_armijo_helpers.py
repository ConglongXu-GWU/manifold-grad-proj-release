import pytest
import torch

from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.projection import feasible_set_projection
from manifold_opt.geometry.retraction import retraction_polar, retraction_qr
from manifold_opt.line_search.armijo import (
    armijo_lhs,
    armijo_residual,
    armijo_rhs_feasible_direction,
    armijo_rhs_projection_arc,
    evaluate_trial_point,
    is_zero_step,
    make_trial_step_feasible_direction,
    make_trial_step_projection_arc,
    resolve_retraction,
    satisfies_armijo,
)
import manifold_opt.line_search.armijo as armijo_module
from manifold_opt.objectives.first_order import WlraFirstOrder
from manifold_opt.objectives.losses import wlra_loss


def _point(dtype=torch.float64):
    U = torch.eye(3, 2, dtype=dtype)
    x = torch.tensor([0.25, -0.5], dtype=dtype)
    V = torch.eye(4, 2, dtype=dtype)
    return U, x, V


def _tangent(point):
    return (
        torch.full_like(point[0], 0.1),
        torch.tensor([0.2, -0.3], dtype=point[1].dtype),
        torch.full_like(point[2], -0.2),
    )


def _data(dtype=torch.float64):
    a = torch.arange(12, dtype=dtype).reshape(3, 4) / 10.0
    w = torch.ones_like(a)
    return a, w


def test_armijo_lhs_computes_loss_decrease():
    f_current = torch.tensor(3.0, dtype=torch.float64)
    f_trial = torch.tensor(1.25, dtype=torch.float64)

    assert torch.allclose(armijo_lhs(f_current, f_trial), torch.tensor(1.75, dtype=torch.float64))


def test_armijo_rhs_feasible_direction_uses_inner_product():
    point = _point()
    grad = _tangent(point)
    projected_direction = (
        0.5 * grad[0],
        torch.tensor([-0.1, 0.4], dtype=torch.float64),
        -0.25 * grad[2],
    )
    alpha = 0.5
    sigma = 0.25

    expected = -sigma * alpha * inner_product(point, grad, projected_direction)

    assert torch.allclose(
        armijo_rhs_feasible_direction(point, grad, projected_direction, alpha, sigma),
        expected,
    )


def test_armijo_rhs_projection_arc_uses_inner_product():
    point = _point()
    grad = _tangent(point)
    projected_step = (
        -0.25 * grad[0],
        torch.tensor([0.3, -0.1], dtype=torch.float64),
        0.75 * grad[2],
    )
    sigma = 0.4

    expected = -sigma * inner_product(point, grad, projected_step)

    assert torch.allclose(armijo_rhs_projection_arc(point, grad, projected_step, sigma), expected)


def test_satisfies_armijo_uses_tolerance():
    assert satisfies_armijo(1.0, 1.0 + 5e-13, tol=1e-12)
    assert not satisfies_armijo(1.0, 1.0 + 2e-12, tol=1e-12)


def test_make_trial_step_feasible_direction_projects_then_scales():
    point = _point()
    grad = _tangent(point)
    s = 0.7
    alpha = 0.25
    r = 0.1

    projected_direction, step = make_trial_step_feasible_direction(point, grad, s, alpha, r)
    expected_projected = feasible_set_projection(point, tuple(-s * block for block in grad), r)

    assert all(torch.allclose(actual, expected) for actual, expected in zip(projected_direction, expected_projected))
    assert all(torch.allclose(actual, alpha * expected) for actual, expected in zip(step, projected_direction))


def test_make_trial_step_projection_arc_projects_current_scaled_gradient():
    point = _point()
    grad = _tangent(point)
    s = 0.6
    r = 0.2

    step = make_trial_step_projection_arc(point, grad, s, r)
    expected = feasible_set_projection(point, tuple(-s * block for block in grad), r)

    assert all(torch.allclose(actual, expected_block) for actual, expected_block in zip(step, expected))


def test_evaluate_trial_point_retracts_then_evaluates_wlra_loss():
    point = _point()
    step = (
        torch.zeros_like(point[0]),
        torch.tensor([0.1, -0.2], dtype=torch.float64),
        torch.zeros_like(point[2]),
    )
    a, w = _data()

    point_trial, f_trial = evaluate_trial_point(point, step, a, w)

    assert torch.allclose(point_trial[1], point[1] + step[1])
    assert torch.allclose(f_trial, wlra_loss(point_trial, a, w))


def test_resolve_retraction_accepts_supported_names():
    assert resolve_retraction("qr") is retraction_qr
    assert resolve_retraction("polar") is retraction_polar


def test_resolve_retraction_rejects_unknown_name():
    with pytest.raises(ValueError, match="retraction"):
        resolve_retraction("cayley")


def test_is_zero_step_detects_zero_and_nonzero_steps():
    point = _point()
    zero_step = tuple(torch.zeros_like(block) for block in point)
    nonzero_step = (
        torch.zeros_like(point[0]),
        torch.tensor([1e-5, 0.0], dtype=torch.float64),
        torch.zeros_like(point[2]),
    )

    assert is_zero_step(zero_step)
    assert not is_zero_step(nonzero_step, tol=1e-12)


def test_is_zero_step_rejects_nonfinite_step():
    point = _point()
    step = tuple(torch.zeros_like(block) for block in point)
    step[1][0] = torch.nan

    with pytest.raises(ValueError):
        is_zero_step(step)


def test_armijo_residual_computes_lhs_minus_rhs():
    lhs = torch.tensor(2.0, dtype=torch.float64)
    rhs = torch.tensor(0.75, dtype=torch.float64)

    assert torch.allclose(armijo_residual(lhs, rhs), torch.tensor(1.25, dtype=torch.float64))


def test_armijo_feasible_direction_uses_fused_current_point_evaluation(monkeypatch):
    point = _point()
    a, w = _data()
    calls = []
    zero_gradient = tuple(torch.zeros_like(block) for block in point)

    def fake_first_order(point_arg, a_arg, w_arg):
        calls.append((point_arg, a_arg, w_arg))
        return WlraFirstOrder(
            data_loss=torch.tensor(1.0, dtype=a_arg.dtype),
            objective=torch.tensor(1.0, dtype=a_arg.dtype),
            riemannian_gradient=zero_gradient,
        )

    monkeypatch.setattr(armijo_module, "wlra_first_order", fake_first_order)
    monkeypatch.setattr(
        armijo_module,
        "wlra_loss",
        lambda *args, **kwargs: pytest.fail("trial loss should not be evaluated for a zero step"),
    )

    result = armijo_module.armijo_feasible_direction(
        point,
        a,
        w,
        r=1.0,
        s=0.1,
        beta=0.5,
        sigma=0.25,
    )

    assert result.accepted
    assert result.reason == "stationary_zero_step"
    assert calls == [(point, a, w)]
