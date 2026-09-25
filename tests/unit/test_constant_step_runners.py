from __future__ import annotations

import json
import sys

import pytest
import torch

import experiments.runners.shared.constant_step_grid as runner
import experiments.runners.constant_step.grid_search_optimize_constraint_constant as constraint_runner
import experiments.runners.constant_step.grid_search_optimize_reg as reg_runner


def _dataset(dtype=torch.float64):
    a_full = torch.arange(12, dtype=dtype).reshape(3, 4) / 10.0
    w = torch.ones_like(a_full)
    w[0, 0] = 0
    a_masked = a_full * w
    return {"M_full": a_full, "M_masked": a_masked, "W": w}


def _initial_point(dtype=torch.float64):
    return (
        torch.eye(3, 2, dtype=dtype),
        torch.tensor([0.2, -0.1], dtype=dtype),
        torch.eye(4, 2, dtype=dtype),
    )


def test_constant_step_wrappers_select_distinct_routines():
    assert reg_runner.ROUTINE == "optimize_reg"
    assert constraint_runner.ROUTINE == "optimize_constraint_constant"


def test_constant_step_grid_uses_learning_rates():
    configs = runner.build_grid(learning_rates=[0.01, 0.1])

    assert configs == [
        runner.ConstantStepConfig(learning_rate=0.01),
        runner.ConstantStepConfig(learning_rate=0.1),
    ]


def test_constant_step_grid_rejects_invalid_learning_rates():
    with pytest.raises(ValueError, match="learning rate"):
        runner.build_grid(learning_rates=[0.0])
    with pytest.raises(ValueError, match="learning rate"):
        runner.parse_csv_learning_rates("")


def test_constant_step_parser_accepts_learning_rates():
    parser = runner.build_arg_parser("optimize_reg", "test")

    args = parser.parse_args(["--learning-rates", "0.01,0.1", "--retraction", "polar"])

    assert args.learning_rates == [0.01, 0.1]
    assert args.retraction == "polar"
    assert args.lmbda == 0.01


def test_constraint_constant_parser_accepts_lmbda_metadata():
    parser = runner.build_arg_parser("optimize_constraint_constant", "test")

    args = parser.parse_args(["--learning-rates", "0.01", "--r", "7", "--lmbda", "1e-4"])

    assert args.r == 7.0
    assert args.lmbda == 1e-4


def test_optimize_reg_runner_writes_data_loss_curve_and_objective_curve(monkeypatch, tmp_path):
    a_full = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 10.0
    w = torch.ones_like(a_full)
    w[0, 0] = 0
    a_masked = a_full * w
    curve_path = tmp_path / "curve.json"
    data_loss_values = iter([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])
    objective_values = iter([torch.tensor(10.0), torch.tensor(20.0), torch.tensor(30.0)])

    monkeypatch.setattr(runner, "grad_reg", lambda point, a, w, lmbda: tuple(torch.zeros_like(block) for block in point))
    monkeypatch.setattr(runner, "wlra_loss", lambda point, a, w: next(data_loss_values))
    monkeypatch.setattr(runner, "wlra_loss_reg", lambda point, a, w, lmbda: next(objective_values))
    monkeypatch.setattr(runner, "rmse_on_missing", lambda point, a, w: torch.tensor(0.5))
    monkeypatch.setattr(runner, "tangent_norm", lambda point, tangent: torch.tensor(0.0))

    record = runner.run_config(
        "optimize_reg",
        runner.ConstantStepConfig(learning_rate=0.01),
        a_full=a_full,
        a_masked=a_masked,
        w=w,
        initial_point=_initial_point(),
        k=2,
        iteration_numbers=2,
        r=100.0,
        lmbda=0.1,
        retraction="qr",
        initialization_method="svd_native",
        curve_path=curve_path,
    )

    curve = json.loads(curve_path.read_text())
    assert record["routine"] == "optimize_reg"
    assert record["final_loss"] == 3.0
    assert record["final_objective"] == 30.0
    assert curve["loss_history"] == [1.0, 2.0]
    assert curve["objective_history"] == [10.0, 20.0]
    assert curve["rmse_history"] == [0.5, 0.5]


def test_constant_step_run_grid_search_initializes_once_and_writes_records(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float32), dataset_path)
    initial_calls = []
    run_calls = []

    def fake_initial_lr(a, k):
        initial_calls.append((a, k))
        return _initial_point(dtype=a.dtype)

    def fake_run_config(routine, config, *, initial_point, initialization_method, curve_path, **kwargs):
        run_calls.append((routine, config, initial_point, initialization_method, curve_path))
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_path.write_text("{}\n")
        return {
            "status": "success",
            "routine": routine,
            "retraction": kwargs["retraction"],
            "config": runner.asdict(config),
            "iteration_numbers": kwargs["iteration_numbers"],
            "rank": kwargs["k"],
            "r": kwargs["r"],
            "lmbda": kwargs["lmbda"],
            "device": "cpu",
            "dtype": "float32",
            "initialization": initialization_method,
            "curve_file": str(curve_path),
            "final_rmse": float(config.learning_rate),
            "final_loss": 1.0,
            "best_rmse": 0.5,
            "best_loss": 0.8,
            "final_gradient_norm": 2.0,
            "final_x_norm": 3.0,
        }

    monkeypatch.setattr(runner, "initial_lr", fake_initial_lr)
    monkeypatch.setattr(runner, "run_config", fake_run_config)

    paths = runner.run_grid_search(
        routine="optimize_constraint_constant",
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raws",
        learning_rates=[0.01, 0.1],
        k=2,
        iteration_numbers=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        run_timestamp="20260527_120000",
    )

    assert len(initial_calls) == 1
    assert len(run_calls) == 2
    assert len(paths) == 2
    assert {json.loads(path.read_text())["config"]["learning_rate"] for path in paths} == {0.01, 0.1}
    assert all(call[2] is run_calls[0][2] for call in run_calls)


def test_constant_step_run_grid_search_writes_failure_record(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.pt"
    torch.save(_dataset(torch.float32), dataset_path)

    def fake_run_config(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(runner, "run_config", fake_run_config)

    paths = runner.run_grid_search(
        routine="optimize_reg",
        dataset_path=dataset_path,
        raw_dir=tmp_path / "raws",
        learning_rates=[0.01],
        k=2,
        iteration_numbers=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        run_timestamp="20260527_120000",
    )

    record = json.loads(paths[0].read_text())
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "failed" in record["error"]
