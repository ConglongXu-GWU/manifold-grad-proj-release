import torch

import manifold_opt.optim.gradient_descent as gradient_descent
from manifold_opt.optim.gradient_descent import (
    optimize_constraint_armijo,
    optimize_constraint_constant,
    optimize_reg,
    optimize_reg_armijo,
)
import manifold_opt.optim.initial as initial
from manifold_opt.optim.initial import initial_lr


def _data():
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10.0
    w = torch.ones_like(a_full)
    return a_full, a_full.clone(), w


def test_initial_lr_returns_v_not_vh():
    a_full, _, _ = _data()

    U, x, V = initial_lr(a_full, k=2)

    assert U.shape == (3, 2)
    assert x.shape == (2,)
    assert V.shape == (4, 2)


def test_initial_lr_preserves_input_dtype_and_device():
    a_full, _, _ = _data()
    a = a_full.to(dtype=torch.float32)

    U, x, V = initial_lr(a, k=2)

    assert U.dtype is torch.float32
    assert x.dtype is torch.float32
    assert V.dtype is torch.float32
    assert U.device == a.device
    assert x.device == a.device
    assert V.device == a.device


def test_initialization_method_reports_native_for_cpu():
    a_full, _, _ = _data()

    assert initial.initialization_method_for_input(a_full) == "svd_native"


def test_initial_lr_routes_mps_input_to_cpu_before_svd(monkeypatch):
    class FakeMpsTensor:
        device = torch.device("mps")
        dtype = torch.float32

        def __init__(self):
            self.cpu_called = False
            self.cpu_tensor = torch.eye(3, 4, dtype=torch.float32)

        def cpu(self):
            self.cpu_called = True
            return self.cpu_tensor

    fake = FakeMpsTensor()
    to_calls = []
    original_svd = torch.linalg.svd

    def fake_svd(value, *, full_matrices):
        assert value is fake.cpu_tensor
        return original_svd(value, full_matrices=full_matrices)

    def fake_to(self, *args, **kwargs):
        to_calls.append((self.shape, kwargs))
        return self

    monkeypatch.setattr(initial.torch.linalg, "svd", fake_svd)
    monkeypatch.setattr(initial.torch.Tensor, "to", fake_to)

    U, x, V = initial.initial_lr(fake, k=2)

    assert fake.cpu_called
    assert U.shape == (3, 2)
    assert x.shape == (2,)
    assert V.shape == (4, 2)
    assert len(to_calls) == 3
    assert all(call[1] == {"device": torch.device("mps"), "dtype": torch.float32} for call in to_calls)


def test_optimize_reg_returns_point_api_blocks_with_zero_iterations():
    a_full, a, w = _data()

    _, _, params, grads = optimize_reg(a_full, a, w, k=2, iteration_numbers=0)

    assert set(params) == {"U", "x", "V"}
    assert set(grads) == {"xi_U", "x_hat", "xi_V"}
    assert params["U"].shape == grads["xi_U"].shape
    assert params["x"].shape == grads["x_hat"].shape
    assert params["V"].shape == grads["xi_V"].shape


def test_optimize_reg_armijo_runs_one_iteration_with_point_api_blocks():
    a_full, a, w = _data()

    losses, rmses, params, grads = optimize_reg_armijo(a_full, a, w, k=2, iteration_numbers=1)

    assert len(losses) == 1
    assert len(rmses) == 1
    assert params["U"].shape == grads["xi_U"].shape
    assert params["x"].shape == grads["x_hat"].shape
    assert params["V"].shape == grads["xi_V"].shape


def test_optimize_constraint_armijo_runs_one_iteration_with_point_api_blocks():
    a_full, a, w = _data()

    losses, rmses, params, grads = optimize_constraint_armijo(a_full, a, w, k=2, r=1.0, iteration_numbers=1)

    assert len(losses) == 1
    assert len(rmses) == 1
    assert params["U"].shape == grads["xi_U"].shape
    assert params["x"].shape == grads["x_hat"].shape
    assert params["V"].shape == grads["xi_V"].shape


def test_armijo_optimizers_final_rmse_mode_records_only_final_rmse():
    a_full, a, w = _data()

    reg_losses, reg_rmses, _, _ = optimize_reg_armijo(a_full, a, w, k=2, iteration_numbers=2, rmse_mode="final")
    constrained_losses, constrained_rmses, _, _ = optimize_constraint_armijo(
        a_full,
        a,
        w,
        k=2,
        r=1.0,
        iteration_numbers=2,
        rmse_mode="final",
    )

    assert len(reg_losses) == 2
    assert len(reg_rmses) == 1
    assert len(constrained_losses) == 2
    assert len(constrained_rmses) == 1


def test_optimize_constraint_constant_runs_one_iteration_with_point_api_blocks():
    a_full, a, w = _data()

    losses, rmses, params, grads = optimize_constraint_constant(a_full, a, w, k=2, r=1.0, iteration_numbers=1)

    assert len(losses) == 1
    assert len(rmses) == 1
    assert params["U"].shape == grads["xi_U"].shape
    assert params["x"].shape == grads["x_hat"].shape
    assert params["V"].shape == grads["xi_V"].shape


def test_constant_optimizers_forward_polar_retraction(monkeypatch):
    a_full, a, w = _data()
    calls = []

    def fake_retraction(point, step):
        calls.append((point, step))
        return point

    monkeypatch.setattr(gradient_descent, "retraction_polar", fake_retraction)

    optimize_reg(a_full, a, w, k=2, iteration_numbers=1, retraction="polar")
    optimize_constraint_constant(a_full, a, w, k=2, r=1.0, iteration_numbers=1, retraction="polar")

    assert len(calls) == 2


def test_constant_optimizers_reject_cayley_retraction():
    a_full, a, w = _data()

    for optimizer in (optimize_reg, optimize_constraint_constant):
        try:
            optimizer(a_full, a, w, k=2, iteration_numbers=1, retraction="cayley")
        except ValueError as exc:
            assert "qr" in str(exc)
            assert "polar" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unsupported constant-step retraction.")
