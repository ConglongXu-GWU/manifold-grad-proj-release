from __future__ import annotations

import json
import sys
from datetime import datetime

import pytest
import torch

import experiments.runners.shared.armijo_grid as runner
import experiments.runners.shared.armijo_layout as layout
import experiments.runners.armijo.grid_search_polar as armijo_polar_runner
import experiments.runners.armijo.grid_search_qr as armijo_qr_runner
import experiments.runners.armijo.run_selected_history_grid as selected_history_runner
import experiments.runners.armijo.feasible_direction.grid_search as feasible_runner
import experiments.runners.armijo.feasible_direction.grid_search_polar as feasible_polar_runner
import experiments.runners.armijo.feasible_direction.grid_search_qr as feasible_qr_runner
import experiments.runners.armijo.projection_arc.grid_search as projection_runner
import experiments.runners.armijo.projection_arc.grid_search_polar as projection_polar_runner
import experiments.runners.armijo.projection_arc.grid_search_qr as projection_qr_runner
import experiments.runners.armijo.wlra_reg.grid_search as reg_runner
import experiments.runners.armijo.wlra_reg.grid_search_polar as reg_polar_runner
import experiments.runners.armijo.wlra_reg.grid_search_qr as reg_qr_runner


def _dataset(dtype=torch.float64):
    a_full = torch.arange(12, dtype=dtype).reshape(3, 4) / 10.0
    w = torch.ones_like(a_full)
    w[0, 0] = 0
    a_masked = a_full * w
    return {"M_full": a_full, "M_masked": a_masked, "W": w}


def test_build_grid_covers_cartesian_product():
    configs = runner.build_grid(
        routines=["armijo_wlra_reg", "armijo_projection_arc"],
        betas=[0.5, 0.8],
        sigmas=[0.25],
        initial_steps=[0.01, 0.1],
        max_backtracks_values=[10],
    )

    assert len(configs) == 8
    assert configs[0] == runner.GridConfig("armijo_wlra_reg", 0.5, 0.25, 0.01, 10)
    assert configs[-1] == runner.GridConfig("armijo_projection_arc", 0.8, 0.25, 0.1, 10)


def test_parse_csv_helpers_reject_invalid_values():
    with pytest.raises(ValueError):
        runner.parse_csv_floats("")
    with pytest.raises(ValueError):
        runner.parse_csv_ints("-1")
    with pytest.raises(ValueError):
        runner.parse_csv_routines("unknown")
    with pytest.raises(ValueError):
        runner.parse_csv_retractions("unknown")


def test_parse_csv_routines_accepts_all_alias():
    assert runner.parse_csv_routines("all") == list(runner.ROUTINES)


def test_parse_csv_retractions_accepts_active_all_alias_and_rejects_cayley():
    assert runner.parse_csv_retractions("all") == ["qr", "polar"]
    with pytest.raises(ValueError, match="cayley"):
        runner.parse_csv_retractions("cayley")


def test_parse_r_auto_and_resolve_radius():
    a_masked = torch.tensor([[3.0, 4.0]], dtype=torch.float64)

    assert runner.parse_r_value("auto") is None
    assert runner.resolve_radius(None, a_masked, 0.25) == pytest.approx(10.0)
    assert runner.resolve_radius(7.0, a_masked, 0.25) == 7.0


def test_armijo_layout_uses_run_id_and_routine_retraction_folders(tmp_path):
    (tmp_path / "260607-01").mkdir()
    assert layout.allocate_run_id(tmp_path, now=datetime(2026, 6, 7)) == "260607-02"
    path = layout.config_path(
        tmp_path / "260607-02",
        routine="armijo_wlra_reg",
        retraction="qr",
        index=3,
    )
    assert path == tmp_path / "260607-02" / "regularized" / "qr" / "regularized_qr_003.json"


def test_per_retraction_armijo_runners_select_retraction():
    assert armijo_qr_runner.RETRACTION == "qr"
    assert armijo_polar_runner.RETRACTION == "polar"


def test_resolve_device_auto_defaults_to_cpu_when_mps_available(monkeypatch):
    monkeypatch.setattr(runner.torch.backends.mps, "is_available", lambda: True)

    assert runner.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_allows_explicit_mps_when_available(monkeypatch):
    monkeypatch.setattr(runner.torch.backends.mps, "is_available", lambda: True)

    assert runner.resolve_device("mps") == torch.device("mps")


def test_single_routine_runners_select_distinct_routines():
    assert reg_runner.ROUTINE == "armijo_wlra_reg"
    assert projection_runner.ROUTINE == "armijo_projection_arc"
    assert feasible_runner.ROUTINE == "armijo_feasible_direction"


def test_routine_retraction_runners_select_distinct_combinations():
    runners = [
        reg_qr_runner,
        reg_polar_runner,
        projection_qr_runner,
        projection_polar_runner,
        feasible_qr_runner,
        feasible_polar_runner,
    ]

    assert {(item.ROUTINE, item.RETRACTION) for item in runners} == {
        ("armijo_wlra_reg", "qr"),
        ("armijo_wlra_reg", "polar"),
        ("armijo_projection_arc", "qr"),
        ("armijo_projection_arc", "polar"),
        ("armijo_feasible_direction", "qr"),
        ("armijo_feasible_direction", "polar"),
    }
    assert projection_qr_runner.MAX_BACKTRACKS == 200
    assert projection_qr_runner.INITIAL_STEPS == (0.1, 0.3, 1.0)
    assert feasible_qr_runner.MAX_BACKTRACKS == 200
    assert feasible_qr_runner.INITIAL_STEPS == (0.1, 0.3, 1.0)
    assert reg_qr_runner.MAX_BACKTRACKS == 200
    assert reg_qr_runner.INITIAL_STEPS == (0.1, 0.3, 1.0)


def test_projection_arc_qr_specialized_grid_uses_fixed_initial_steps():
    configs = projection_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5])

    assert len(configs) == 6
    assert [config.initial_step for config in configs] == [0.1, 0.3, 1.0, 0.1, 0.3, 1.0]


def test_projection_arc_qr_specialized_grid_accepts_custom_initial_steps():
    configs = projection_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5], initial_steps=[0.02, 0.2])

    assert len(configs) == 4
    assert [config.initial_step for config in configs] == [0.02, 0.2, 0.02, 0.2]


def test_feasible_direction_qr_specialized_grid_uses_fixed_initial_steps():
    configs = feasible_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5])

    assert len(configs) == 6
    assert [config.initial_step for config in configs] == [0.1, 0.3, 1.0, 0.1, 0.3, 1.0]


def test_feasible_direction_qr_specialized_grid_accepts_custom_initial_steps():
    configs = feasible_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5], initial_steps=[0.2, 2.0])

    assert len(configs) == 4
    assert [config.initial_step for config in configs] == [0.2, 2.0, 0.2, 2.0]


def test_wlra_reg_qr_specialized_grid_uses_fixed_initial_steps():
    configs = reg_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5])

    assert len(configs) == 6
    assert [config.initial_step for config in configs] == [0.1, 0.3, 1.0, 0.1, 0.3, 1.0]


def test_wlra_reg_qr_specialized_grid_accepts_custom_initial_steps():
    configs = reg_qr_runner.build_grid(betas=[0.5], sigmas=[0.25, 0.5], initial_steps=[0.2, 2.0])

    assert len(configs) == 4
    assert [config.initial_step for config in configs] == [0.2, 2.0, 0.2, 2.0]


def test_qr_specialized_grids_reject_nonpositive_initial_steps():
    for module in (projection_qr_runner, feasible_qr_runner, reg_qr_runner):
        with pytest.raises(ValueError, match="initial step"):
            module.build_grid(betas=[0.5], sigmas=[0.25], initial_steps=[0.0])


def test_qr_specialized_parsers_accept_initial_steps_from_cli(monkeypatch):
    for module in (projection_qr_runner, feasible_qr_runner, reg_qr_runner):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                module.__file__,
                "--betas",
                "0.5",
                "--sigmas",
                "0.25",
                "--initial-steps",
                "0.02,0.2",
            ],
        )

        args = module.parse_args()

        assert args.initial_steps == [0.02, 0.2]


def test_qr_specialized_parsers_default_to_final_rmse_mode(monkeypatch):
    for module in (projection_qr_runner, feasible_qr_runner, reg_qr_runner):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                module.__file__,
                "--betas",
                "0.5",
                "--sigmas",
                "0.25",
            ],
        )

        args = module.parse_args()

        assert args.rmse_mode == "final"


def test_projection_arc_qr_runner_initializes_once_for_all_configs(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float32), dataset_path)
    initial_calls = []
    run_calls = []

    def fake_initial_lr(a, k):
        initial_calls.append((a, k))
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_run_config(config, *, initial_point, initialization_method, **kwargs):
        run_calls.append((config, initial_point, initialization_method))
        return {
            "status": "success",
            "routine": projection_qr_runner.ROUTINE,
            "retraction": projection_qr_runner.RETRACTION,
            "config": projection_qr_runner.asdict(config) | {"max_backtracks": projection_qr_runner.MAX_BACKTRACKS},
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "armijo_tol": kwargs["armijo_tol"],
            "device": "cpu",
            "dtype": "float32",
            "initialization": initialization_method,
            "final_rmse": float(config.initial_step),
            "final_loss": 1.0,
            "final_gradient_norm": 2.0,
            "final_x_norm": 3.0,
            "max_backtracks_hit": False,
            "final_backtracks": 0,
        }

    monkeypatch.setattr(projection_qr_runner, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(projection_qr_runner, "run_config", fake_run_config)

    paths = projection_qr_runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raws",
        betas=[0.5],
        sigmas=[0.25, 0.5],
        k=2,
        iteration_numbers=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        run_timestamp="20260524_120000",
    )

    assert len(initial_calls) == 1
    assert len(run_calls) == 6
    assert len(paths) == 6
    assert {call[2] for call in run_calls} == {"svd_native"}
    first_initial_point = run_calls[0][1]
    assert all(call[1] is first_initial_point for call in run_calls)


def test_feasible_direction_qr_runner_initializes_once_for_all_configs(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float32), dataset_path)
    initial_calls = []
    run_calls = []

    def fake_initial_lr(a, k):
        initial_calls.append((a, k))
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_run_config(config, *, initial_point, initialization_method, **kwargs):
        run_calls.append((config, initial_point, initialization_method))
        return {
            "status": "success",
            "routine": feasible_qr_runner.ROUTINE,
            "retraction": feasible_qr_runner.RETRACTION,
            "config": feasible_qr_runner.asdict(config) | {"max_backtracks": feasible_qr_runner.MAX_BACKTRACKS},
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "armijo_tol": kwargs["armijo_tol"],
            "device": "cpu",
            "dtype": "float32",
            "initialization": initialization_method,
            "final_rmse": float(config.initial_step),
            "final_loss": 1.0,
            "final_gradient_norm": 2.0,
            "final_x_norm": 3.0,
            "max_backtracks_hit": False,
            "final_backtracks": 0,
        }

    monkeypatch.setattr(feasible_qr_runner, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(feasible_qr_runner, "run_config", fake_run_config)

    paths = feasible_qr_runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raws",
        betas=[0.5],
        sigmas=[0.25, 0.5],
        k=2,
        iteration_numbers=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        run_timestamp="20260524_120000",
    )

    assert len(initial_calls) == 1
    assert len(run_calls) == 6
    assert len(paths) == 6
    assert {call[2] for call in run_calls} == {"svd_native"}
    first_initial_point = run_calls[0][1]
    assert all(call[1] is first_initial_point for call in run_calls)


def test_wlra_reg_qr_runner_initializes_once_for_all_configs(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float32), dataset_path)
    initial_calls = []
    run_calls = []

    def fake_initial_lr(a, k):
        initial_calls.append((a, k))
        return (
            torch.eye(a.shape[0], k, dtype=a.dtype),
            torch.arange(1, k + 1, dtype=a.dtype),
            torch.eye(a.shape[1], k, dtype=a.dtype),
        )

    def fake_run_config(config, *, initial_point, initialization_method, **kwargs):
        run_calls.append((config, initial_point, initialization_method))
        return {
            "status": "success",
            "routine": reg_qr_runner.ROUTINE,
            "retraction": reg_qr_runner.RETRACTION,
            "config": reg_qr_runner.asdict(config) | {"max_backtracks": reg_qr_runner.MAX_BACKTRACKS},
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "armijo_tol": kwargs["armijo_tol"],
            "device": "cpu",
            "dtype": "float32",
            "initialization": initialization_method,
            "final_rmse": float(config.initial_step),
            "final_loss": 1.0,
            "final_data_loss": 0.9,
            "final_gradient_norm": 2.0,
            "final_x_norm": 3.0,
            "max_backtracks_hit": False,
            "final_backtracks": 0,
        }

    monkeypatch.setattr(reg_qr_runner, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(reg_qr_runner, "run_config", fake_run_config)

    paths = reg_qr_runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raws",
        betas=[0.5],
        sigmas=[0.25, 0.5],
        k=2,
        iteration_numbers=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        run_timestamp="20260524_120000",
    )

    assert len(initial_calls) == 1
    assert len(run_calls) == 6
    assert len(paths) == 6
    assert {call[2] for call in run_calls} == {"svd_native"}
    first_initial_point = run_calls[0][1]
    assert all(call[1] is first_initial_point for call in run_calls)


def test_feasible_direction_qr_run_config_calls_armijo_with_fixed_qr_parameters(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full * w
    initial_point = (
        torch.eye(3, 2, dtype=torch.float64),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        torch.eye(4, 2, dtype=torch.float64),
    )
    calls = []

    class Result:
        accepted = True
        point_next = initial_point
        m = 17
        reason = "accepted"

    def fake_armijo(point, a, w_arg, *, r, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        calls.append((point, a, w_arg, r, s, beta, sigma, max_backtracks, armijo_tol, retraction))
        return Result()

    monkeypatch.setattr(feasible_qr_runner, "armijo_feasible_direction", fake_armijo)
    monkeypatch.setattr(feasible_qr_runner, "grad", lambda point, a, w_arg: tuple(torch.zeros_like(block) for block in point))
    monkeypatch.setattr(feasible_qr_runner, "wlra_loss", lambda point, a, w_arg: torch.tensor(1.0, dtype=a.dtype))
    monkeypatch.setattr(feasible_qr_runner, "rmse_on_missing", lambda point, a, w_arg: torch.tensor(0.2, dtype=a.dtype))
    monkeypatch.setattr(feasible_qr_runner, "tangent_norm", lambda point, tangent: torch.tensor(0.3, dtype=point[1].dtype))

    config = feasible_qr_runner.FeasibleDirectionQrConfig(beta=0.9, sigma=0.75, initial_step=0.1)
    record = feasible_qr_runner.run_config(
        config,
        a_full=a_full,
        a_masked=a_masked,
        w=w,
        initial_point=initial_point,
        k=2,
        iteration_numbers=1,
        r=7.0,
        lmbda=0.001,
        armijo_tol=1e-12,
        initialization_method="svd_native",
    )

    assert len(calls) == 1
    assert calls[0][3:] == (7.0, 0.1, 0.9, 0.75, 200, 1e-12, "qr")
    assert record["routine"] == "armijo_feasible_direction"
    assert record["retraction"] == "qr"
    assert record["metric_mode"] == "final"
    assert "best_rmse" not in record
    assert record["final_backtracks"] == 17


def test_feasible_direction_qr_run_config_history_mode_records_best_rmse(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full * w
    initial_point = (
        torch.eye(3, 2, dtype=torch.float64),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        torch.eye(4, 2, dtype=torch.float64),
    )

    class Result:
        accepted = True
        point_next = initial_point
        m = 0
        reason = "accepted"

    monkeypatch.setattr(feasible_qr_runner, "armijo_feasible_direction", lambda *args, **kwargs: Result())
    monkeypatch.setattr(feasible_qr_runner, "grad", lambda point, a, w_arg: tuple(torch.zeros_like(block) for block in point))
    monkeypatch.setattr(feasible_qr_runner, "wlra_loss", lambda point, a, w_arg: torch.tensor(1.0, dtype=a.dtype))
    rmse_values = iter([torch.tensor(0.5, dtype=a_full.dtype), torch.tensor(0.2, dtype=a_full.dtype)])
    monkeypatch.setattr(feasible_qr_runner, "rmse_on_missing", lambda point, a, w_arg: next(rmse_values))
    monkeypatch.setattr(feasible_qr_runner, "tangent_norm", lambda point, tangent: torch.tensor(0.3, dtype=point[1].dtype))

    record = feasible_qr_runner.run_config(
        feasible_qr_runner.FeasibleDirectionQrConfig(beta=0.9, sigma=0.75, initial_step=0.1),
        a_full=a_full,
        a_masked=a_masked,
        w=w,
        initial_point=initial_point,
        k=2,
        iteration_numbers=2,
        r=7.0,
        lmbda=0.001,
        armijo_tol=1e-12,
        initialization_method="svd_native",
        rmse_mode="history",
    )

    assert record["metric_mode"] == "history"
    assert record["final_rmse"] == 0.2
    assert record["best_rmse"] == 0.2
    assert record["rmse_history"] == [0.5, 0.2]


def test_wlra_reg_qr_run_config_calls_armijo_with_fixed_qr_parameters(monkeypatch):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    w = torch.ones_like(a_full)
    a_masked = a_full * w
    initial_point = (
        torch.eye(3, 2, dtype=torch.float64),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        torch.eye(4, 2, dtype=torch.float64),
    )
    calls = []

    class Result:
        accepted = True
        point_next = initial_point
        m = 17
        reason = "accepted"

    def fake_armijo(point, a, w_arg, *, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction):
        calls.append((point, a, w_arg, lmbda, s, beta, sigma, max_backtracks, armijo_tol, retraction))
        return Result()

    monkeypatch.setattr(reg_qr_runner, "armijo_wlra_reg", fake_armijo)
    monkeypatch.setattr(
        reg_qr_runner,
        "grad_reg",
        lambda point, a, w_arg, lmbda: tuple(torch.zeros_like(block) for block in point),
    )
    monkeypatch.setattr(reg_qr_runner, "wlra_loss", lambda point, a, w_arg: torch.tensor(0.9, dtype=a.dtype))
    monkeypatch.setattr(
        reg_qr_runner,
        "wlra_loss_reg",
        lambda point, a, w_arg, lmbda: torch.tensor(1.0, dtype=a.dtype),
    )
    monkeypatch.setattr(reg_qr_runner, "rmse_on_missing", lambda point, a, w_arg: torch.tensor(0.2, dtype=a.dtype))
    monkeypatch.setattr(reg_qr_runner, "tangent_norm", lambda point, tangent: torch.tensor(0.3, dtype=point[1].dtype))

    config = reg_qr_runner.WlraRegQrConfig(beta=0.9, sigma=0.75, initial_step=0.1)
    record = reg_qr_runner.run_config(
        config,
        a_full=a_full,
        a_masked=a_masked,
        w=w,
        initial_point=initial_point,
        k=2,
        iteration_numbers=1,
        r=7.0,
        lmbda=0.001,
        armijo_tol=1e-12,
        initialization_method="svd_native",
    )

    assert len(calls) == 1
    assert calls[0][3:] == (0.001, 0.1, 0.9, 0.75, 200, 1e-12, "qr")
    assert record["routine"] == "armijo_wlra_reg"
    assert record["retraction"] == "qr"
    assert record["metric_mode"] == "final"
    assert "best_rmse" not in record
    assert record["final_loss"] == 1.0
    assert record["final_data_loss"] == 0.9
    assert record["final_backtracks"] == 17


def test_single_routine_parser_omits_routines_argument():
    parser = runner.build_arg_parser(include_routines=False, default_run_name="single_grid")

    args = parser.parse_args(
        [
            "--betas",
            "0.5",
            "--sigmas",
            "0.25",
            "--initial-steps",
            "0.01",
            "--max-backtracks",
            "7",
        ]
    )

    assert args.run_name == "single_grid"
    assert args.device == "cpu"
    assert args.rmse_mode == "final"
    assert not hasattr(args, "routines")


def test_fixed_routine_retraction_parser_omits_both_arguments():
    parser = runner.build_arg_parser(
        include_routines=False,
        include_retraction=False,
        default_run_name="fixed_grid",
    )

    args = parser.parse_args(
        [
            "--betas",
            "0.5",
            "--sigmas",
            "0.25",
            "--initial-steps",
            "0.01",
            "--max-backtracks",
            "7",
        ]
    )

    assert args.run_name == "fixed_grid"
    assert args.rmse_mode == "final"
    assert not hasattr(args, "routines")
    assert not hasattr(args, "retraction")


def test_fixed_max_backtracks_parser_omits_max_backtracks_argument():
    parser = runner.build_arg_parser(
        include_routines=False,
        include_retraction=False,
        include_max_backtracks=False,
        default_run_name="fixed_backtracks_grid",
    )

    args = parser.parse_args(
        [
            "--betas",
            "0.5",
            "--sigmas",
            "0.25",
            "--initial-steps",
            "0.01",
        ]
    )

    assert args.run_name == "fixed_backtracks_grid"
    assert args.rmse_mode == "final"
    assert not hasattr(args, "routines")
    assert not hasattr(args, "retraction")
    assert not hasattr(args, "max_backtracks")


def test_run_grid_search_writes_records_summary_and_resumes(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    calls = []

    def fake_reg_history(config, *, a_masked, initial_point, **kwargs):
        point = runner.clone_initial_fn(initial_point)(a_masked, kwargs["k"])
        calls.append(("reg", config.beta, config.sigma, config.max_backtracks, config.initial_step, point))
        return [torch.tensor(1.5, dtype=a_masked.dtype)], [torch.tensor(3.0, dtype=a_masked.dtype)], [
            torch.tensor(0.3, dtype=a_masked.dtype)
        ], point, [3]

    def fake_constraint(
        a_full,
        a,
        w,
        *,
        k,
        initial_fn,
        armijo_rule,
        beta,
        sigma,
        max_backtracks,
        lr,
        diagnostics=None,
        **kwargs,
    ):
        point = initial_fn(a, k)
        calls.append((armijo_rule, beta, sigma, max_backtracks, lr, point))
        rmse = 0.1 if armijo_rule == "projection_arc" else 0.2
        if diagnostics is not None:
            diagnostics["armijo_evaluation_history"] = [4 if armijo_rule == "projection_arc" else 5]
        return [torch.tensor(2.0, dtype=a.dtype)], [torch.tensor(rmse, dtype=a.dtype)], {
            "U": point[0],
            "x": point[1],
            "V": point[2],
        }, {}

    monkeypatch.setattr(runner, "_run_regularized_fair_history", fake_reg_history)
    monkeypatch.setattr(runner, "optimize_constraint_armijo", fake_constraint)

    summary = runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        figures_dir=tmp_path / "figures",
        run_name="test_grid",
        routines=runner.ROUTINES,
        betas=[0.5],
        sigmas=[0.25],
        initial_steps=[0.01],
        max_backtracks_values=[7],
        k=2,
        iteration_numbers=1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    raw_files = sorted(path for path in (tmp_path / "raw" / "test_grid").rglob("*.json") if path.name != "grid_config.json")
    assert len(raw_files) == 3
    first_record = json.loads(raw_files[0].read_text())
    records = [json.loads(path.read_text()) for path in raw_files]
    records_by_routine = {record["config"]["routine"]: record for record in records}
    assert first_record["metric_mode"] == "final"
    assert "best_rmse" not in first_record
    assert "final_wlra_loss" in first_record
    assert records_by_routine["armijo_wlra_reg"]["armijo_evaluation_history"] == [3]
    assert records_by_routine["armijo_wlra_reg"]["mean_armijo_evaluations"] == 3.0
    assert records_by_routine["armijo_wlra_reg"]["final_armijo_evaluations"] == 3
    assert records_by_routine["armijo_wlra_reg"]["max_armijo_evaluations"] == 3
    assert records_by_routine["armijo_projection_arc"]["armijo_evaluation_history"] == [4]
    assert records_by_routine["armijo_feasible_direction"]["armijo_evaluation_history"] == [5]
    grid_config = json.loads((tmp_path / "raw" / "test_grid" / "grid_config.json").read_text())
    assert grid_config["metric_mode"] == "final"
    assert grid_config["r_policy"] == "fixed"
    assert len(summary) == 3
    best_configs = tmp_path / "processed" / "test_grid" / "best_configs.csv"
    assert best_configs.exists()
    assert "mean_armijo_evaluations" in best_configs.read_text()
    assert (tmp_path / "processed" / "test_grid" / "grid_config.json").exists()
    assert (tmp_path / "processed" / "test_grid" / "reproducibility.md").exists()
    assert (tmp_path / "figures" / "test_grid" / "test_grid_best_rmse_by_routine.png").stat().st_size > 0
    assert (tmp_path / "figures" / "test_grid" / "test_grid_rmse_by_config.png").stat().st_size > 0
    assert [call[0] for call in calls] == ["reg", "projection_arc", "feasible_direction"]
    first_point = calls[0][-1]
    assert all(torch.equal(first_point[idx], calls[1][-1][idx]) for idx in range(3))
    assert all(first_point[idx].data_ptr() != calls[1][-1][idx].data_ptr() for idx in range(3))

    calls.clear()
    runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        figures_dir=tmp_path / "figures",
        run_name="test_grid",
        routines=runner.ROUTINES,
        betas=[0.5],
        sigmas=[0.25],
        initial_steps=[0.01],
        max_backtracks_values=[7],
        k=2,
        iteration_numbers=1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert calls == []


def test_run_grid_search_sweeps_rank_lambda_and_retraction_with_auto_r(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float64), dataset_path)
    seen = []

    def fake_run_config(config, **kwargs):
        seen.append((kwargs["k"], kwargs["lmbda"], kwargs["r"], kwargs["retraction"], config.routine))
        return {
            "config_id": runner.config_id(
                config,
                retraction=kwargs["retraction"],
                rank=kwargs["k"],
                lmbda=kwargs["lmbda"],
            ),
            "status": "success",
            "config": runner.asdict(config),
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "armijo_tol": kwargs["armijo_tol"],
            "retraction": kwargs["retraction"],
            "metric_mode": kwargs["rmse_mode"],
            "device": "cpu",
            "dtype": "float64",
            "final_objective": 3.0,
            "final_wlra_loss": 3.0,
            "final_data_loss": 3.0,
            "final_rmse": 0.5,
            "best_objective": 3.0,
            "best_wlra_loss": 3.0,
        }

    monkeypatch.setattr(runner, "run_config", fake_run_config)

    summary = runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        figures_dir=tmp_path / "figures",
        run_name="sweep_grid",
        routines=["armijo_wlra_reg"],
        betas=[0.5],
        sigmas=[0.25],
        initial_steps=[0.01],
        max_backtracks_values=[200],
        ks=[2, 3],
        lmbdas=[1.0, 0.25],
        retractions=["qr", "polar"],
        r=None,
        iteration_numbers=1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert len(seen) == 8
    assert len(summary) == 8
    assert {item[3] for item in seen} == {"qr", "polar"}
    assert {item[0] for item in seen} == {2, 3}
    assert {item[1] for item in seen} == {1.0, 0.25}
    assert all(item[2] > 0 for item in seen)


def test_run_grid_search_writes_failure_checkpoint_and_continues(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)

    def fail_run_config(*args, **kwargs):
        error = RuntimeError("boom")
        error.armijo_evaluation_history = [1, 2]
        raise error

    monkeypatch.setattr(runner, "run_config", fail_run_config)

    summary = runner.run_grid_search(
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        run_name="failed_grid",
        routines=["armijo_wlra_reg"],
        betas=[0.5],
        sigmas=[0.25],
        initial_steps=[0.01],
        max_backtracks_values=[7],
        k=2,
        iteration_numbers=1,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    raw_file = next(path for path in (tmp_path / "raw" / "failed_grid").rglob("*.json") if path.name != "grid_config.json")
    record = json.loads(raw_file.read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["error"] == "boom"
    assert record["armijo_evaluation_history"] == [1, 2]
    assert record["mean_armijo_evaluations"] == 1.5
    assert record["final_armijo_evaluations"] == 2
    assert record["max_armijo_evaluations"] == 2
    assert summary == []
    assert (tmp_path / "processed" / "failed_grid" / "best_configs.csv").exists()


def _source_record(*, routine, retraction="qr", rank=2, lmbda=1.0, rmse=0.2, loss=1.0, step=0.1):
    config = {
        "routine": routine,
        "beta": 0.5,
        "sigma": 0.25,
        "initial_step": step,
        "max_backtracks": 200,
    }
    return {
        "status": "success",
        "config": config,
        "config_id": f"{routine}_{step}",
        "iteration_numbers": 200,
        "rank": rank,
        "r": 7.0,
        "lmbda": lmbda,
        "armijo_tol": 1e-12,
        "retraction": retraction,
        "metric_mode": "final",
        "device": "cpu",
        "dtype": "float64",
        "final_objective": loss,
        "final_wlra_loss": loss,
        "final_data_loss": loss,
        "final_rmse": rmse,
        "best_objective": loss,
        "best_wlra_loss": loss,
    }


def _write_source(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def test_selected_history_candidates_rank_by_rmse_then_loss(tmp_path):
    raw_root = tmp_path / "raw"
    base = raw_root / "source_qr_regularized" / "regularized" / "qr"
    _write_source(base / "a.json", _source_record(routine="armijo_wlra_reg", rmse=0.2, loss=2.0, step=0.1))
    _write_source(base / "b.json", _source_record(routine="armijo_wlra_reg", rmse=0.1, loss=5.0, step=0.2))
    _write_source(base / "c.json", _source_record(routine="armijo_wlra_reg", rmse=0.1, loss=1.0, step=0.3))

    candidates = selected_history_runner.load_source_candidates(raw_root, source_run_prefix="source")
    ranked = selected_history_runner.rank_source_candidates(
        candidates,
        routine="armijo_wlra_reg",
        retraction="qr",
        ranks=[2],
        lmbdas=[1.0],
    )

    group = ("qr", 2, 1.0, "armijo_wlra_reg")
    assert [item.config.initial_step for item in ranked[group]] == [0.3, 0.2, 0.1]
    assert [item.candidate_rank for item in ranked[group]] == [1, 2, 3]


def test_selected_history_runner_falls_back_and_writes_manifest(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    raw_root = tmp_path / "raw"
    base = raw_root / "source_qr_regularized" / "regularized" / "qr"
    _write_source(base / "best.json", _source_record(routine="armijo_wlra_reg", rmse=0.1, loss=1.0, step=0.1))
    _write_source(base / "second.json", _source_record(routine="armijo_wlra_reg", rmse=0.2, loss=1.5, step=0.2))

    def fake_run_config(config, **kwargs):
        if config.initial_step == 0.1:
            error = RuntimeError("max_backtracks_exceeded")
            error.armijo_evaluation_history = [201]
            raise error
        return {
            "config_id": runner.config_id(
                config,
                retraction=kwargs["retraction"],
                rank=kwargs["k"],
                lmbda=kwargs["lmbda"],
            ),
            "status": "success",
            "config": runner.asdict(config) if hasattr(runner, "asdict") else {
                "routine": config.routine,
                "beta": config.beta,
                "sigma": config.sigma,
                "initial_step": config.initial_step,
                "max_backtracks": config.max_backtracks,
            },
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "armijo_tol": kwargs["armijo_tol"],
            "retraction": kwargs["retraction"],
            "metric_mode": kwargs["rmse_mode"],
            "device": "cpu",
            "dtype": "float64",
            "final_objective": 3.0,
            "final_wlra_loss": 3.0,
            "final_data_loss": 3.0,
            "final_rmse": 0.3,
            "best_objective": 3.0,
            "best_wlra_loss": 3.0,
            "best_rmse": 0.3,
            "rmse_history": [0.4, 0.3],
            "wlra_loss_history": [4.0, 3.0],
            "armijo_evaluation_history": [2, 2],
            "mean_armijo_evaluations": 2.0,
            "final_armijo_evaluations": 2,
            "max_armijo_evaluations": 2,
        }

    monkeypatch.setattr(selected_history_runner, "run_config", fake_run_config)

    selected_history_runner.run_selected_history_grid(
        dataset_path=dataset_path,
        source_run_prefix="source",
        run_name="history",
        routine="armijo_wlra_reg",
        retraction="qr",
        raw_dir=raw_root,
        processed_dir=tmp_path / "processed",
        ranks=[2],
        lmbdas=[1.0],
        iteration_numbers=2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    records = [
        json.loads(path.read_text())
        for path in (raw_root / "history").rglob("*.json")
        if path.name != "grid_config.json"
    ]
    assert [record["status"] for record in records] == ["failed", "success"]
    manifest = (tmp_path / "processed" / "history" / "selected_history_attempts.csv").read_text()
    assert "max_backtracks_exceeded" in manifest
    assert "True" in manifest
    assert (tmp_path / "processed" / "history" / "best_configs.csv").exists()


def test_selected_history_runner_skips_existing_success(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    raw_root = tmp_path / "raw"
    source_base = raw_root / "source_qr_regularized" / "regularized" / "qr"
    _write_source(source_base / "best.json", _source_record(routine="armijo_wlra_reg", rmse=0.1, loss=1.0))
    existing_base = raw_root / "history" / "regularized" / "qr"
    _write_source(
        existing_base / "regularized_qr_001.json",
        {
            **_source_record(routine="armijo_wlra_reg", rank=2, lmbda=1.0, rmse=0.3, loss=3.0),
            "run_id": "history",
            "config_index": 1,
            "iteration_numbers": 2,
            "metric_mode": "history",
            "best_rmse": 0.3,
            "rmse_history": [0.4, 0.3],
            "wlra_loss_history": [4.0, 3.0],
            "source_config_id": "armijo_wlra_reg_0.1",
            "source_candidate_rank": 1,
        },
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_config should not be called for an existing success")

    monkeypatch.setattr(selected_history_runner, "run_config", fail_if_called)

    selected_history_runner.run_selected_history_grid(
        dataset_path=dataset_path,
        source_run_prefix="source",
        run_name="history",
        routine="armijo_wlra_reg",
        retraction="qr",
        raw_dir=raw_root,
        processed_dir=tmp_path / "processed",
        ranks=[2],
        lmbdas=[1.0],
        iteration_numbers=2,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert (tmp_path / "processed" / "history" / "selected_history_attempts.csv").exists()


def test_selected_history_runner_raises_when_all_candidates_fail(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(), dataset_path)
    raw_root = tmp_path / "raw"
    base = raw_root / "source_qr_regularized" / "regularized" / "qr"
    _write_source(base / "best.json", _source_record(routine="armijo_wlra_reg", rmse=0.1, loss=1.0, step=0.1))
    _write_source(base / "second.json", _source_record(routine="armijo_wlra_reg", rmse=0.2, loss=2.0, step=0.2))

    monkeypatch.setattr(selected_history_runner, "run_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="No successful history records"):
        selected_history_runner.run_selected_history_grid(
            dataset_path=dataset_path,
            source_run_prefix="source",
            run_name="history",
            routine="armijo_wlra_reg",
            retraction="qr",
            raw_dir=raw_root,
            processed_dir=tmp_path / "processed",
            ranks=[2],
            lmbdas=[1.0],
            iteration_numbers=2,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

    records = [
        json.loads(path.read_text())
        for path in (raw_root / "history").rglob("*.json")
        if path.name != "grid_config.json"
    ]
    assert [record["status"] for record in records] == ["failed", "failed"]
