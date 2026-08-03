import pytest
import torch

import manifold_opt.optim.gradient_descent as gradient_descent
from manifold_opt.line_search.armijo import ArmijoResult
from manifold_opt.optim.gradient_descent import optimize_reg_armijo


def _data():
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10.0
    w = torch.ones_like(a_full)
    return a_full, a_full.clone(), w


def _initial(a, k):
    return (
        torch.eye(a.shape[0], k, dtype=a.dtype, device=a.device),
        torch.tensor([0.2, -0.1], dtype=a.dtype, device=a.device),
        torch.eye(a.shape[1], k, dtype=a.dtype, device=a.device),
    )


def _result(point, *, s, armijo_evaluations=0):
    step = tuple(torch.full_like(block, 0.01) for block in point)
    point_next = tuple(block + step_block for block, step_block in zip(point, step))
    grad = tuple(torch.full_like(block, 0.2) for block in point)
    f_current = torch.tensor(3.0, dtype=point[1].dtype)
    f_next = torch.tensor(2.5, dtype=point[1].dtype)
    return ArmijoResult(
        accepted=True,
        point_next=point_next,
        step=step,
        m=0,
        alpha=1.0,
        s=s,
        f_current=f_current,
        f_next=f_next,
        armijo_lhs=f_current - f_next,
        armijo_rhs=torch.tensor(0.1, dtype=point[1].dtype),
        descent_inner=torch.tensor(-0.4, dtype=point[1].dtype),
        grad=grad,
        reason="accepted",
        armijo_evaluations=armijo_evaluations,
    )


def test_optimize_reg_armijo_uses_armijo_wlra_reg(monkeypatch):
    a_full, a, w = _data()
    calls = []

    def fake_armijo(point, a_arg, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        calls.append((point, a_arg, w_arg, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction))
        return _result(point, s=s)

    monkeypatch.setattr(gradient_descent, "armijo_wlra_reg", fake_armijo)

    losses, _, params, grads = optimize_reg_armijo(
        a_full,
        a,
        w,
        k=2,
        lmbda=0.3,
        lr=0.04,
        iteration_numbers=1,
        initial_fn=_initial,
        beta=0.4,
        sigma=0.2,
        max_backtracks=7,
        armijo_tol=1e-10,
    )

    assert len(calls) == 1
    assert calls[0][3:] == (0.3, 0.04, 0.4, 0.2, 7, 1e-10, "qr")
    assert torch.allclose(losses[0], torch.tensor(3.0, dtype=torch.float64))
    assert all(torch.allclose(params[name], calls[0][0][idx] + 0.01) for idx, name in enumerate(("U", "x", "V")))
    assert all(torch.allclose(grads[name], torch.full_like(grads[name], 0.2)) for name in ("xi_U", "x_hat", "xi_V"))


def test_optimize_reg_armijo_records_armijo_evaluation_diagnostics(monkeypatch):
    a_full, a, w = _data()
    calls = []

    def fake_armijo(point, a_arg, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        calls.append(point)
        return _result(point, s=s, armijo_evaluations=len(calls) + 2)

    monkeypatch.setattr(gradient_descent, "armijo_wlra_reg", fake_armijo)
    diagnostics = {}

    optimize_reg_armijo(
        a_full,
        a,
        w,
        k=2,
        iteration_numbers=2,
        initial_fn=_initial,
        diagnostics=diagnostics,
    )

    assert diagnostics["armijo_evaluation_history"] == [3, 4]


def test_optimize_reg_armijo_forwards_retraction(monkeypatch):
    a_full, a, w = _data()
    calls = []

    def fake_armijo(point, a_arg, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        calls.append(retraction)
        return _result(point, s=s)

    monkeypatch.setattr(gradient_descent, "armijo_wlra_reg", fake_armijo)

    optimize_reg_armijo(
        a_full,
        a,
        w,
        k=2,
        iteration_numbers=1,
        retraction="polar",
        initial_fn=_initial,
    )

    assert calls == ["polar"]


def test_optimize_reg_armijo_rejects_unsupported_retraction():
    a_full, a, w = _data()

    with pytest.raises(ValueError, match="qr.*polar"):
        optimize_reg_armijo(a_full, a, w, k=2, iteration_numbers=1, retraction="unsupported", initial_fn=_initial)


def test_optimize_reg_armijo_raises_when_armijo_step_fails(monkeypatch):
    a_full, a, w = _data()

    def fake_armijo(point, a_arg, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        result = _result(point, s=s)
        return ArmijoResult(
            accepted=False,
            point_next=result.point_next,
            step=result.step,
            m=result.m,
            alpha=result.alpha,
            s=result.s,
            f_current=result.f_current,
            f_next=result.f_next,
            armijo_lhs=result.armijo_lhs,
            armijo_rhs=result.armijo_rhs,
            descent_inner=result.descent_inner,
            grad=result.grad,
            reason="max_backtracks_exceeded",
        )

    monkeypatch.setattr(gradient_descent, "armijo_wlra_reg", fake_armijo)

    try:
        optimize_reg_armijo(a_full, a, w, k=2, iteration_numbers=1, initial_fn=_initial)
    except RuntimeError as exc:
        assert "max_backtracks_exceeded" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for failed Armijo step.")
