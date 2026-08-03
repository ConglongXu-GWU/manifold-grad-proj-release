import pytest
import torch

import manifold_opt.line_search.armijo as armijo
from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.riemannian_gradient import grad_reg
from manifold_opt.line_search.armijo import armijo_wlra_reg
from manifold_opt.objectives.losses import wlra_loss_reg


def _point(dtype=torch.float64):
    U = torch.eye(3, 2, dtype=dtype)
    x = torch.tensor([0.2, -0.1], dtype=dtype)
    V = torch.eye(4, 2, dtype=dtype)
    return U, x, V


def _data(dtype=torch.float64):
    a = torch.arange(12, dtype=dtype).reshape(3, 4) / 10.0
    w = torch.ones_like(a)
    return a, w


def _reconstruction(point):
    U, x, V = point
    return (U * x.unsqueeze(0)) @ V.transpose(0, 1)


def test_armijo_wlra_reg_accepts_first_trial():
    point = _point()
    a, w = _data()
    lmbda = 0.1
    s = 1e-6

    result = armijo_wlra_reg(point, a, w, lmbda=lmbda, s=s, beta=0.5, sigma=0.25)
    gradient = grad_reg(point, a, w, lmbda)
    direction = tuple(-s * block for block in gradient)

    assert result.accepted
    assert result.m == 0
    assert result.alpha == 1.0
    assert result.s == s
    assert result.armijo_evaluations == 1
    assert all(torch.allclose(actual, expected) for actual, expected in zip(result.step, direction))
    assert result.armijo_lhs + 1e-12 >= result.armijo_rhs


def test_armijo_wlra_reg_accepts_with_each_retraction():
    point = _point()
    a, w = _data()

    for retraction in ("qr", "polar"):
        result = armijo_wlra_reg(
            point,
            a,
            w,
            lmbda=0.1,
            s=1e-6,
            beta=0.5,
            sigma=0.25,
            retraction=retraction,
        )

        assert result.accepted
        assert result.m == 0
        assert torch.allclose(result.point_next[0].transpose(0, 1) @ result.point_next[0], torch.eye(2, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(result.point_next[2].transpose(0, 1) @ result.point_next[2], torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_armijo_wlra_reg_backtracks_on_alpha(monkeypatch):
    point = _point()
    a, w = _data()
    decisions = iter([False, True])

    monkeypatch.setattr(armijo, "satisfies_armijo", lambda lhs, rhs, tol=1e-12: next(decisions))

    result = armijo_wlra_reg(point, a, w, lmbda=0.1, s=1e-3, beta=0.5, sigma=0.25)

    assert result.accepted
    assert result.m == 1
    assert result.alpha == 0.5
    assert result.armijo_evaluations == 2


def test_armijo_wlra_reg_rejects_unsupported_retraction():
    point = _point()
    a, w = _data()

    with pytest.raises(ValueError, match="qr.*polar"):
        armijo_wlra_reg(point, a, w, lmbda=0.1, s=0.2, beta=0.5, sigma=0.25, retraction="unsupported")


def test_armijo_wlra_reg_zero_step_returns_stationary_result():
    point = _point()
    a = _reconstruction(point)
    w = torch.ones_like(a)

    result = armijo_wlra_reg(point, a, w, lmbda=0.0, s=1.0, beta=0.5, sigma=0.25)

    assert result.accepted
    assert result.m == 0
    assert result.reason == "stationary_zero_step"
    assert result.armijo_evaluations == 0
    assert all(torch.count_nonzero(block) == 0 for block in result.step)
    assert result.point_next is point
    assert torch.allclose(result.f_next, wlra_loss_reg(point, a, w, 0.0))


def test_armijo_wlra_reg_returns_failure_when_max_backtracks_exceeded(monkeypatch):
    point = _point()
    a, w = _data()

    monkeypatch.setattr(armijo, "satisfies_armijo", lambda lhs, rhs, tol=1e-12: False)

    result = armijo_wlra_reg(
        point,
        a,
        w,
        lmbda=0.1,
        s=1e-3,
        beta=0.5,
        sigma=0.25,
        max_backtracks=1,
    )

    assert not result.accepted
    assert result.reason == "max_backtracks_exceeded"
    assert result.m == 1
    assert result.armijo_evaluations == 2


def test_armijo_wlra_reg_uses_inner_product_rhs():
    point = _point()
    a, w = _data()
    lmbda = 0.1
    s = 1e-6

    result = armijo_wlra_reg(point, a, w, lmbda=lmbda, s=s, beta=0.5, sigma=0.25)
    gradient = grad_reg(point, a, w, lmbda)
    direction = tuple(-s * block for block in gradient)
    expected_inner = inner_product(point, gradient, direction)

    assert torch.allclose(result.descent_inner, expected_inner)
    assert torch.allclose(result.armijo_rhs, -0.25 * result.alpha * expected_inner)


def test_armijo_wlra_reg_does_not_use_feasible_projection(monkeypatch):
    point = _point()
    a, w = _data()

    def fail_projection(*args, **kwargs):
        raise AssertionError("armijo_wlra_reg must not call feasible_set_projection.")

    monkeypatch.setattr(armijo, "feasible_set_projection", fail_projection)

    result = armijo_wlra_reg(point, a, w, lmbda=0.1, s=1e-6, beta=0.5, sigma=0.25)

    assert result.accepted
