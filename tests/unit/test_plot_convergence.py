from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import torch

import experiments.plotting.shared.convergence_common as convergence_common
import experiments.plotting.armijo.curves.plot_convergence_grid as plot_convergence_grid
import experiments.plotting.armijo.curves.plot_convergence_polar as plot_convergence_polar
import experiments.plotting.armijo.curves.plot_convergence_qr as plot_convergence_qr
import experiments.summaries.armijo.summarize_projection_arc_qr_grid as summarize_projection_arc_qr_grid
from manifold_opt.line_search.armijo import ArmijoResult


def _dataset(dtype=torch.float32):
    a_full = torch.arange(12, dtype=dtype).reshape(3, 4)
    w = torch.ones_like(a_full)
    w[0, 0] = 0
    a_masked = a_full * w
    return {"M_full": a_full, "M_masked": a_masked, "W": w}


def test_load_mnist_masked_dataset_success(tmp_path):
    path = tmp_path / "dataset.pt"
    torch.save(_dataset(), path)

    a_full, a_masked, w = convergence_common.load_mnist_masked_dataset(path)

    assert a_full.dtype == torch.float32
    assert a_masked.dtype == torch.float32
    assert w.dtype == torch.float32
    assert a_full.shape == (3, 4)
    assert torch.equal(w, _dataset(torch.float32)["W"])


def test_load_mnist_masked_dataset_allows_explicit_float64_cpu(tmp_path):
    path = tmp_path / "dataset.pt"
    torch.save(_dataset(), path)

    a_full, a_masked, w = convergence_common.load_mnist_masked_dataset(path, dtype="float64", device="cpu")

    assert a_full.dtype == torch.float64
    assert a_masked.dtype == torch.float64
    assert w.dtype == torch.float64


def test_resolve_experiment_runtime_defaults_to_float32_cpu_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(convergence_common.torch.backends.mps, "is_available", lambda: False)

    device, dtype = convergence_common.resolve_experiment_runtime()

    assert device == torch.device("cpu")
    assert dtype is torch.float32


def test_resolve_experiment_runtime_defaults_to_cpu_when_mps_available(monkeypatch):
    monkeypatch.setattr(convergence_common.torch.backends.mps, "is_available", lambda: True)

    device, dtype = convergence_common.resolve_experiment_runtime()

    assert device == torch.device("cpu")
    assert dtype is torch.float32


def test_resolve_experiment_runtime_auto_selects_cpu_when_mps_available(monkeypatch):
    monkeypatch.setattr(convergence_common.torch.backends.mps, "is_available", lambda: True)

    device, dtype = convergence_common.resolve_experiment_runtime(device="auto")

    assert device == torch.device("cpu")
    assert dtype is torch.float32


def test_resolve_experiment_runtime_allows_explicit_mps_when_available(monkeypatch):
    monkeypatch.setattr(convergence_common.torch.backends.mps, "is_available", lambda: True)

    device, dtype = convergence_common.resolve_experiment_runtime(device="mps")

    assert device == torch.device("mps")
    assert dtype is torch.float32


def test_resolve_experiment_runtime_rejects_float64_mps(monkeypatch):
    monkeypatch.setattr(convergence_common.torch.backends.mps, "is_available", lambda: True)

    try:
        convergence_common.resolve_experiment_runtime(device="mps", dtype="float64")
    except ValueError as exc:
        assert "float64" in str(exc)
    else:
        raise AssertionError("Expected ValueError for float64 on MPS.")


def test_load_mnist_masked_dataset_rejects_missing_keys(tmp_path):
    path = tmp_path / "dataset.pt"
    data = _dataset()
    data.pop("W")
    torch.save(data, path)

    try:
        convergence_common.load_mnist_masked_dataset(path)
    except ValueError as exc:
        assert "missing required key" in str(exc)
        assert "W" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing W.")


def test_load_mnist_masked_dataset_rejects_shape_mismatch(tmp_path):
    path = tmp_path / "dataset.pt"
    data = _dataset()
    data["W"] = torch.ones(2, 4)
    torch.save(data, path)

    try:
        convergence_common.load_mnist_masked_dataset(path)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes.")


def test_load_mnist_masked_dataset_rejects_non_binary_weights(tmp_path):
    path = tmp_path / "dataset.pt"
    data = _dataset()
    data["W"][0, 1] = 0.5
    torch.save(data, path)

    try:
        convergence_common.load_mnist_masked_dataset(path)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-binary W.")


def test_run_armijo_comparison_uses_both_rules_and_shared_initialization(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full.clone()
    calls = []

    def fake_initial_lr(a, k):
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_optimize(a_full_arg, a_arg, w_arg, *, k, initial_fn, armijo_rule, retraction, **kwargs):
        point = initial_fn(a_arg, k)
        calls.append((armijo_rule, retraction, point))
        losses = [torch.tensor(10.0 if armijo_rule == "projection_arc" else 9.0, dtype=torch.float64)]
        rmses = [torch.tensor(1.0 if armijo_rule == "projection_arc" else 0.9, dtype=torch.float64)]
        return losses, rmses, {}, {}

    monkeypatch.setattr(convergence_common, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(convergence_common, "optimize_constraint_armijo", fake_optimize)

    history = convergence_common.run_armijo_comparison(
        a_full,
        a_masked,
        w,
        retraction="qr",
        k=2,
        iteration_numbers=1,
    )

    assert [call[0] for call in calls] == ["projection_arc", "feasible_direction"]
    assert [call[1] for call in calls] == ["qr", "qr"]
    first_point = calls[0][2]
    second_point = calls[1][2]
    assert all(torch.equal(left, right) for left, right in zip(first_point, second_point))
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first_point, second_point))
    assert history == {
        "projection_arc": {"loss": [10.0], "rmse": [1.0]},
        "feasible_direction": {"loss": [9.0], "rmse": [0.9]},
    }


def test_retraction_script_constants_are_distinct():
    assert plot_convergence_qr.RETRACTION == "qr"
    assert plot_convergence_polar.RETRACTION == "polar"
    assert plot_convergence_grid.REPO_ROOT == plot_convergence_qr.REPO_ROOT


def test_regularized_armijo_records_unregularized_loss(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    w[0, 0] = 0
    a_masked = a_full * w
    initial_point = (
        torch.eye(a_full.shape[0], 2, dtype=torch.float64),
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        torch.eye(a_full.shape[1], 2, dtype=torch.float64),
    )
    loss_calls = []
    armijo_calls = []

    def fake_wlra_loss(point, a_arg, w_arg):
        loss_calls.append(point)
        return torch.tensor(10.0 + len(loss_calls), dtype=torch.float64)

    def fake_armijo(point, a_arg, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        armijo_calls.append((lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction))
        point_next = tuple(block + 0.01 for block in point)
        return ArmijoResult(
            accepted=True,
            point_next=point_next,
            step=tuple(torch.full_like(block, 0.01) for block in point),
            m=0,
            alpha=1.0,
            s=s,
            f_current=torch.tensor(999.0, dtype=torch.float64),
            f_next=torch.tensor(998.0, dtype=torch.float64),
            armijo_lhs=torch.tensor(1.0, dtype=torch.float64),
            armijo_rhs=torch.tensor(0.1, dtype=torch.float64),
            descent_inner=torch.tensor(-1.0, dtype=torch.float64),
            grad=tuple(torch.zeros_like(block) for block in point),
            reason="accepted",
        )

    monkeypatch.setattr(convergence_common, "wlra_loss", fake_wlra_loss)
    monkeypatch.setattr(convergence_common, "armijo_wlra_reg", fake_armijo)

    history = convergence_common.run_regularized_armijo_fair_loss(
        a_full,
        a_masked,
        w,
        initial_point=initial_point,
        k=2,
        lmbda=0.3,
        lr=0.04,
        iteration_numbers=2,
        beta=0.4,
        sigma=0.2,
        max_backtracks=7,
        armijo_tol=1e-10,
        retraction="qr",
    )

    assert history["loss"] == [11.0, 12.0]
    assert len(history["rmse"]) == 2
    assert armijo_calls == [(0.3, 0.04, 0.4, 0.2, 7, 1e-10, "qr")] * 2


def test_run_qr_three_method_comparison_includes_regularized_method(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full.clone()
    optimize_calls = []

    def fake_initial_lr(a, k):
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_optimize(a_full_arg, a_arg, w_arg, *, armijo_rule, retraction, **kwargs):
        optimize_calls.append((armijo_rule, retraction))
        loss = torch.tensor(1.0 if armijo_rule == "feasible_direction" else 2.0, dtype=torch.float64)
        rmse = torch.tensor(0.1 if armijo_rule == "feasible_direction" else 0.2, dtype=torch.float64)
        return [loss], [rmse], {}, {}

    def fake_regularized(*args, **kwargs):
        assert kwargs["lmbda"] == 0.3
        assert kwargs["retraction"] == "qr"
        return {"loss": [3.0], "rmse": [0.3]}

    monkeypatch.setattr(convergence_common, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(convergence_common, "optimize_constraint_armijo", fake_optimize)
    monkeypatch.setattr(convergence_common, "run_regularized_armijo_fair_loss", fake_regularized)

    history = convergence_common.run_qr_three_method_comparison(
        a_full,
        a_masked,
        w,
        k=2,
        lmbda=0.3,
        iteration_numbers=1,
    )

    assert optimize_calls == [("feasible_direction", "qr"), ("projection_arc", "qr")]
    assert history == {
        "feasible_direction": {"loss": [1.0], "rmse": [0.1]},
        "projection_arc": {"loss": [2.0], "rmse": [0.2]},
        "regularized_armijo": {"loss": [3.0], "rmse": [0.3]},
    }


def test_run_three_method_comparison_supports_polar_retraction(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full.clone()
    optimize_calls = []

    def fake_initial_lr(a, k):
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_optimize(a_full_arg, a_arg, w_arg, *, armijo_rule, retraction, **kwargs):
        optimize_calls.append((armijo_rule, retraction))
        loss = torch.tensor(1.0 if armijo_rule == "feasible_direction" else 2.0, dtype=torch.float64)
        rmse = torch.tensor(0.1 if armijo_rule == "feasible_direction" else 0.2, dtype=torch.float64)
        return [loss], [rmse], {}, {}

    def fake_regularized(*args, **kwargs):
        assert kwargs["retraction"] == "polar"
        return {"loss": [3.0], "rmse": [0.3]}

    monkeypatch.setattr(convergence_common, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(convergence_common, "optimize_constraint_armijo", fake_optimize)
    monkeypatch.setattr(convergence_common, "run_regularized_armijo_fair_loss", fake_regularized)

    history = convergence_common.run_three_method_comparison(
        a_full,
        a_masked,
        w,
        retraction="polar",
        k=2,
        iteration_numbers=1,
    )

    assert optimize_calls == [("feasible_direction", "polar"), ("projection_arc", "polar")]
    assert history == {
        "feasible_direction": {"loss": [1.0], "rmse": [0.1]},
        "projection_arc": {"loss": [2.0], "rmse": [0.2]},
        "regularized_armijo": {"loss": [3.0], "rmse": [0.3]},
    }


def test_plot_helpers_create_png_files(tmp_path):
    history = {
        "projection_arc": {"loss": [3.0, 2.0], "rmse": [1.0, 0.8]},
        "feasible_direction": {"loss": [3.0, 1.5], "rmse": [1.0, 0.7]},
        "regularized_armijo": {"loss": [3.0, 1.8], "rmse": [1.0, 0.75]},
    }
    loss_path = tmp_path / "loss.png"
    rmse_path = tmp_path / "rmse.png"

    convergence_common.plot_loss_curves(history, loss_path)
    convergence_common.plot_rmse_curves(history, rmse_path)

    assert loss_path.exists()
    assert loss_path.stat().st_size > 0
    assert rmse_path.exists()
    assert rmse_path.stat().st_size > 0


def test_convergence_title_reports_experiment_parameters():
    title = convergence_common.convergence_title(
        "Loss",
        retraction="polar",
        lmbda=1e-4,
        r=14647.17624406148,
        beta=0.9,
        sigma=0.75,
    )

    assert title == "Loss convergence (retraction=polar, lambda=0.0001, r=14647.2, beta=0.9, sigma=0.75)"


def test_create_retraction_convergence_plots_writes_timestamped_figures(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    history = {
        "projection_arc": {"loss": [3.0, 2.0], "rmse": [1.0, 0.8]},
        "feasible_direction": {"loss": [3.0, 1.5], "rmse": [1.0, 0.7]},
    }

    calls = []

    def fake_run(a_full_arg, a_arg, w_arg, *, retraction, **kwargs):
        calls.append(retraction)
        return history

    monkeypatch.setattr(convergence_common, "run_armijo_comparison", fake_run)

    output_dir = tmp_path / "figures"
    convergence_common.create_retraction_convergence_plots(
        "polar",
        dataset_path=dataset_path,
        output_dir=output_dir,
        device="cpu",
        run_date=date(2026, 5, 18),
    )

    expected_names = {
        "armijo_polar_loss_convergence_18_05_2026.png",
        "armijo_polar_rmse_convergence_18_05_2026.png",
    }
    actual_names = {path.name for path in output_dir.iterdir()}

    assert calls == ["polar"]
    assert actual_names == expected_names
    assert all((output_dir / name).stat().st_size > 0 for name in expected_names)


def test_create_qr_three_method_convergence_plots_writes_timestamped_figures(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    history = {
        "feasible_direction": {"loss": [3.0, 2.0], "rmse": [1.0, 0.8]},
        "projection_arc": {"loss": [3.0, 1.5], "rmse": [1.0, 0.7]},
        "regularized_armijo": {"loss": [3.0, 1.8], "rmse": [1.0, 0.75]},
    }
    calls = []
    titles = []

    def fake_run(a_full_arg, a_arg, w_arg, *, lmbda, **kwargs):
        calls.append(lmbda)
        return history

    original_plot_loss_curves = convergence_common.plot_loss_curves
    original_plot_rmse_curves = convergence_common.plot_rmse_curves

    def capture_loss_title(history_arg, output_path, title=None):
        titles.append(title)
        original_plot_loss_curves(history_arg, output_path, title=title)

    def capture_rmse_title(history_arg, output_path, title=None):
        titles.append(title)
        original_plot_rmse_curves(history_arg, output_path, title=title)

    monkeypatch.setattr(convergence_common, "run_qr_three_method_comparison", fake_run)
    monkeypatch.setattr(convergence_common, "plot_loss_curves", capture_loss_title)
    monkeypatch.setattr(convergence_common, "plot_rmse_curves", capture_rmse_title)

    output_dir = tmp_path / "figures"
    convergence_common.create_qr_three_method_convergence_plots(
        dataset_path=dataset_path,
        output_dir=output_dir,
        lmbda=0.3,
        r=7.0,
        beta=0.4,
        sigma=0.2,
        device="cpu",
        run_date=date(2026, 5, 18),
    )

    expected_names = {
        "armijo_qr_loss_convergence_18_05_2026.png",
        "armijo_qr_rmse_convergence_18_05_2026.png",
    }
    actual_names = {path.name for path in output_dir.iterdir()}

    assert calls == [0.3]
    assert titles == [
        "Loss convergence (retraction=qr, lambda=0.3, r=7, beta=0.4, sigma=0.2)",
        "RMSE convergence (retraction=qr, lambda=0.3, r=7, beta=0.4, sigma=0.2)",
    ]
    assert actual_names == expected_names
    assert all((output_dir / name).stat().st_size > 0 for name in expected_names)


def test_create_polar_three_method_convergence_plots_writes_timestamped_figures(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    history = {
        "feasible_direction": {"loss": [3.0, 2.0], "rmse": [1.0, 0.8]},
        "projection_arc": {"loss": [3.0, 1.5], "rmse": [1.0, 0.7]},
        "regularized_armijo": {"loss": [3.0, 1.8], "rmse": [1.0, 0.75]},
    }
    calls = []
    titles = []

    def fake_run(a_full_arg, a_arg, w_arg, *, retraction, lmbda, **kwargs):
        calls.append((retraction, lmbda))
        return history

    original_plot_loss_curves = convergence_common.plot_loss_curves
    original_plot_rmse_curves = convergence_common.plot_rmse_curves

    def capture_loss_title(history_arg, output_path, title=None):
        titles.append(title)
        original_plot_loss_curves(history_arg, output_path, title=title)

    def capture_rmse_title(history_arg, output_path, title=None):
        titles.append(title)
        original_plot_rmse_curves(history_arg, output_path, title=title)

    monkeypatch.setattr(convergence_common, "run_three_method_comparison", fake_run)
    monkeypatch.setattr(convergence_common, "plot_loss_curves", capture_loss_title)
    monkeypatch.setattr(convergence_common, "plot_rmse_curves", capture_rmse_title)

    output_dir = tmp_path / "figures"
    convergence_common.create_polar_three_method_convergence_plots(
        dataset_path=dataset_path,
        output_dir=output_dir,
        lmbda=0.3,
        r=7.0,
        beta=0.4,
        sigma=0.2,
        device="cpu",
        run_date=date(2026, 5, 18),
    )

    expected_names = {
        "armijo_polar_loss_convergence_18_05_2026.png",
        "armijo_polar_rmse_convergence_18_05_2026.png",
    }
    actual_names = {path.name for path in output_dir.iterdir()}

    assert calls == [("polar", 0.3)]
    assert titles == [
        "Loss convergence (retraction=polar, lambda=0.3, r=7, beta=0.4, sigma=0.2)",
        "RMSE convergence (retraction=polar, lambda=0.3, r=7, beta=0.4, sigma=0.2)",
    ]
    assert actual_names == expected_names
    assert all((output_dir / name).stat().st_size > 0 for name in expected_names)


def _history_record(retraction, routine, rank, lmbda, rmse, loss):
    return {
        "status": "success",
        "metric_mode": "history",
        "retraction": retraction,
        "rank": rank,
        "lmbda": lmbda,
        "config": {"routine": routine, "beta": 0.5, "sigma": 0.25, "initial_step": 0.1},
        "final_rmse": rmse,
        "final_wlra_loss": loss,
        "wlra_loss_history": [loss + 2.0, loss + 1.0, loss],
        "regularized_objective_history": [100.0, 90.0, 80.0],
        "rmse_history": [rmse + 0.2, rmse + 0.1, rmse],
    }


def test_select_best_curve_records_uses_final_rmse_then_fair_loss():
    records = [
        _history_record("qr", "armijo_wlra_reg", 32, 1e-4, 0.2, 9.0),
        _history_record("qr", "armijo_wlra_reg", 32, 1e-4, 0.2, 7.0),
    ]

    selected = convergence_common.select_best_curve_records(records, retraction="qr", ranks=[32], lmbdas=[1e-4])

    assert selected[(32, 1e-4, "regularized_armijo")]["final_wlra_loss"] == 7.0


def test_winning_curve_method_uses_final_rmse_then_fair_loss():
    selected = {
        (32, 1e-4, "feasible_direction"): _history_record("qr", "armijo_feasible_direction", 32, 1e-4, 0.2, 9.0),
        (32, 1e-4, "projection_arc"): _history_record("qr", "armijo_projection_arc", 32, 1e-4, 0.2, 7.0),
        (32, 1e-4, "regularized_armijo"): _history_record("qr", "armijo_wlra_reg", 32, 1e-4, 0.3, 4.0),
    }

    assert convergence_common.winning_curve_method(selected, rank=32, lmbda=1e-4) == "projection_arc"


def test_load_curve_records_accepts_multiple_raw_dirs(tmp_path):
    raw_a = tmp_path / "raw_a"
    raw_b = tmp_path / "raw_b"
    (raw_a / "regularized" / "qr").mkdir(parents=True)
    (raw_b / "projection_arc" / "qr").mkdir(parents=True)
    (raw_a / "regularized" / "qr" / "a.json").write_text(
        summarize_projection_arc_qr_grid.json.dumps(_history_record("qr", "armijo_wlra_reg", 32, 1e-4, 0.2, 7.0))
    )
    (raw_b / "projection_arc" / "qr" / "b.json").write_text(
        summarize_projection_arc_qr_grid.json.dumps(
            _history_record("qr", "armijo_projection_arc", 32, 1e-4, 0.3, 8.0)
        )
    )

    records = convergence_common.load_curve_records([raw_a, raw_b], retraction="qr")

    assert len(records) == 2
    assert {record["config"]["routine"] for record in records} == {"armijo_wlra_reg", "armijo_projection_arc"}


def test_plot_retraction_grid_curves_writes_figure_from_history_records(tmp_path):
    records = []
    for rank in (32, 64, 128):
        for lmbda in (1e-2, 1e-4, 1e-6):
            for routine, rmse in (
                ("armijo_feasible_direction", 0.3),
                ("armijo_projection_arc", 0.2),
                ("armijo_wlra_reg", 0.1),
            ):
                records.append(_history_record("qr", routine, rank, lmbda, rmse, 10.0 + rmse))

    output = convergence_common.plot_retraction_grid_curves(
        records,
        retraction="qr",
        metric="loss",
        output_path=tmp_path / "qr_loss.png",
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_create_grid_convergence_figures_rmse_only_from_multiple_raw_dirs(tmp_path):
    raw_a = tmp_path / "raw_a"
    raw_b = tmp_path / "raw_b"
    (raw_a / "regularized" / "qr").mkdir(parents=True)
    (raw_b / "projection_arc" / "qr").mkdir(parents=True)
    for rank in (32, 64, 128):
        for lmbda in (1e-2, 1e-4, 1e-6):
            for routine, rmse, raw_dir in (
                ("armijo_feasible_direction", 0.3, raw_a),
                ("armijo_projection_arc", 0.2, raw_b),
                ("armijo_wlra_reg", 0.1, raw_a),
            ):
                label = routine.replace("armijo_", "")
                target = raw_dir / label / "qr"
                target.mkdir(parents=True, exist_ok=True)
                (target / f"{rank}_{lmbda}_{routine}.json").write_text(
                    summarize_projection_arc_qr_grid.json.dumps(_history_record("qr", routine, rank, lmbda, rmse, 10.0))
                )

    paths = convergence_common.create_grid_convergence_figures(
        raw_dirs=[raw_a, raw_b],
        output_dir=tmp_path / "figures",
        retractions=["qr"],
        metrics=["rmse"],
        run_name="unit",
    )

    assert [path.name for path in paths] == ["armijo_qr_rmse_grid_convergence_unit.png"]
    assert paths[0].exists()
    assert paths[0].stat().st_size > 0


def test_plot_merged_grid_convergence_curves_stacks_labeled_datasets(tmp_path):
    datasets = []
    for dataset_label, offset in (("Sampled MNIST", 0.0), ("Sampled CIFAR-10", 0.1)):
        records = []
        for retraction in ("qr", "polar"):
            for routine, rmse in (
                ("armijo_feasible_direction", 0.3 + offset),
                ("armijo_projection_arc", 0.2 + offset),
                ("armijo_wlra_reg", 0.1 + offset),
            ):
                records.append(
                    _history_record(retraction, routine, 32, 1e-2, rmse, 10.0 + rmse)
                )
        datasets.append((dataset_label, records))

    output = convergence_common.plot_merged_grid_convergence_curves(
        datasets,
        retractions=["qr", "polar"],
        metric="rmse",
        output_path=tmp_path / "merged.png",
        ranks=[32],
        lmbdas=[1e-2],
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_selected_grid_uses_manuscript_rank_symbol():
    records = [
        _history_record("qr", "armijo_wlra_reg", 32, 1e-2, 0.1, 10.0),
    ]
    selected = convergence_common.select_best_curve_records(
        records,
        retraction="qr",
        ranks=[32],
        lmbdas=[1e-2],
    )
    fig, axes = plt.subplots(1, 1, squeeze=False)

    convergence_common._plot_selected_grid(
        axes,
        selected,
        metric="rmse",
        rank_values=[32],
        lmbda_values=[1e-2],
    )

    assert axes[0][0].get_title() == "p=32, lambda=0.01"
    plt.close(fig)


def test_plot_selected_grid_applies_publication_font_size():
    records = [
        _history_record("qr", "armijo_wlra_reg", 32, 1e-2, 0.1, 10.0),
    ]
    selected = convergence_common.select_best_curve_records(
        records,
        retraction="qr",
        ranks=[32],
        lmbdas=[1e-2],
    )
    fig, axes = plt.subplots(1, 1, squeeze=False)

    convergence_common._plot_selected_grid(
        axes,
        selected,
        metric="rmse",
        rank_values=[32],
        lmbda_values=[1e-2],
        font_size=34.0,
        publication_layout=True,
    )

    ax = axes[0][0]
    assert ax.title.get_fontsize() == 34.0
    assert ax.xaxis.label.get_fontsize() == 34.0
    assert ax.yaxis.label.get_fontsize() == 34.0
    assert {label.get_fontsize() for label in ax.get_xticklabels()} == {34.0}
    assert {text.get_fontsize() for text in ax.get_legend().get_texts()} == {34.0}
    plt.close(fig)


def test_projection_arc_qr_summary_markdown_highlights_best_config(tmp_path):
    raw_dir = tmp_path / "raws"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    records = [
        {
            "status": "success",
            "routine": "armijo_projection_arc",
            "retraction": "qr",
            "config": {"beta": 0.5, "sigma": 0.25, "initial_step": 0.001, "max_backtracks": 200},
            "r": 7.0,
            "final_rmse": 0.4,
            "final_loss": 12.0,
            "final_gradient_norm": 1.5,
            "final_x_norm": 2.5,
            "max_backtracks_hit": False,
            "final_backtracks": 3,
            "device": "mps:0",
            "run_timestamp": "20260524_010101",
            "config_index": 1,
        },
        {
            "status": "success",
            "routine": "armijo_projection_arc",
            "retraction": "qr",
            "config": {"beta": 0.9, "sigma": 0.5, "initial_step": 0.01, "max_backtracks": 200},
            "r": 7.0,
            "final_rmse": 0.2,
            "final_loss": 10.0,
            "final_gradient_norm": 0.5,
            "final_x_norm": 3.5,
            "max_backtracks_hit": True,
            "final_backtracks": 7,
            "device": "mps:0",
            "run_timestamp": "20260524_010101",
            "config_index": 2,
        },
    ]
    for index, record in enumerate(records):
        (raw_dir / f"armijo_projection_arc_qr_test_{index}.json").write_text(
            summarize_projection_arc_qr_grid.json.dumps(record)
        )

    output = summarize_projection_arc_qr_grid.write_summary(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        timestamp="20260524_020202",
    )

    text = output.read_text()
    assert output.name == "projection_arc_qr_summary.md"
    assert "- Device: gpu/mps" in text
    assert (
        "| beta | sigma | s-bar | r | lambda | max-backtracks | retraction | final rmse | final loss | "
        "final data loss | final gradient norm | final x norm | max-backtracks hit | approx runtime | final backtracks |"
    ) in text
    assert "| **0.9** | **0.5** | **0.01** | **7** |  | **200** | **qr** | **0.2** |" in text
    assert "**true**" in text
    assert "**7**" in text
