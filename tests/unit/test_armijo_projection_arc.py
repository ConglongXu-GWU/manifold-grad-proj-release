import pytest
import torch

import manifold_opt.line_search.armijo as armijo
from manifold_opt.geometry.projection import feasible_set_projection
from manifold_opt.geometry.riemannian_gradient import grad
from manifold_opt.line_search.armijo import armijo_projection_arc
from manifold_opt.objectives.losses import wlra_loss


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


def test_projection_arc_accepts_first_trial():
    point = _point()
    a, w = _data()
    s_bar = 1e-6
    r = 10.0

    result = armijo_projection_arc(point, a, w, r=r, s_bar=s_bar, beta=0.5, sigma=0.25)
    expected_step = feasible_set_projection(point, tuple(-s_bar * block for block in grad(point, a, w)), r)

    assert result.accepted
    assert result.m == 0
    assert result.alpha == 1.0
    assert result.s == s_bar
    assert result.armijo_evaluations == 1
    assert all(torch.allclose(actual, expected) for actual, expected in zip(result.step, expected_step))
    assert result.armijo_lhs + 1e-12 >= result.armijo_rhs


def test_projection_arc_accepts_with_each_retraction():
    point = _point()
    a, w = _data()

    for retraction in ("qr", "polar"):
        result = armijo_projection_arc(
            point,
            a,
            w,
            r=10.0,
            s_bar=1e-6,
            beta=0.5,
            sigma=0.25,
            retraction=retraction,
        )

        assert result.accepted
        assert result.m == 0
        assert torch.allclose(result.point_next[0].transpose(0, 1) @ result.point_next[0], torch.eye(2, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(result.point_next[2].transpose(0, 1) @ result.point_next[2], torch.eye(2, dtype=torch.float64), atol=1e-12)


def test_projection_arc_backtracks_on_s_and_reprojects(monkeypatch):
    point = _point()
    a, w = _data()
    projection_calls = 0
    decisions = iter([False, True])
    original_projection = armijo.feasible_set_projection

    def counted_projection(point, tangent, r):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(point, tangent, r)

    monkeypatch.setattr(armijo, "feasible_set_projection", counted_projection)
    monkeypatch.setattr(armijo, "satisfies_armijo", lambda lhs, rhs, tol=1e-12: next(decisions))

    result = armijo_projection_arc(point, a, w, r=10.0, s_bar=1e-3, beta=0.5, sigma=0.25)
    expected_step = feasible_set_projection(point, tuple(-result.s * block for block in grad(point, a, w)), 10.0)

    assert result.accepted
    assert result.m == 1
    assert result.alpha == 1.0
    assert result.s == 0.5e-3
    assert result.armijo_evaluations == 2
    assert projection_calls == 2
    assert all(torch.allclose(actual, expected) for actual, expected in zip(result.step, expected_step))


def test_projection_arc_rejects_cayley_retraction():
    point = _point()
    a, w = _data()

    with pytest.raises(ValueError, match="qr.*polar"):
        armijo_projection_arc(point, a, w, r=10.0, s_bar=0.2, beta=0.5, sigma=0.25, retraction="cayley")


def test_projection_arc_zero_step_returns_stationary_result():
    point = _point()
    a = _reconstruction(point)
    w = torch.ones_like(a)

    result = armijo_projection_arc(point, a, w, r=10.0, s_bar=1.0, beta=0.5, sigma=0.25)

    assert result.accepted
    assert result.m == 0
    assert result.s == 1.0
    assert result.reason == "stationary_zero_step"
    assert result.armijo_evaluations == 0
    assert all(torch.count_nonzero(block) == 0 for block in result.step)
    assert result.point_next is point
    assert torch.allclose(result.f_next, wlra_loss(point, a, w))


def test_projection_arc_returns_failure_when_max_backtracks_exceeded(monkeypatch):
    point = _point()
    a, w = _data()

    monkeypatch.setattr(armijo, "satisfies_armijo", lambda lhs, rhs, tol=1e-12: False)

    result = armijo_projection_arc(
        point,
        a,
        w,
        r=10.0,
        s_bar=1e-3,
        beta=0.5,
        sigma=0.25,
        max_backtracks=1,
    )

    assert not result.accepted
    assert result.reason == "max_backtracks_exceeded"
    assert result.m == 1
    assert result.armijo_evaluations == 2
