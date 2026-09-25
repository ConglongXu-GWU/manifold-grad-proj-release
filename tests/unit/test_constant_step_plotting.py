from __future__ import annotations

import json
from pathlib import Path

import experiments.plotting.constant_step.curves.plot_constant_step_grid as plot_constant_step_grid
import experiments.summaries.constant_step.summarize_constant_step_grid as summarize_constant_step_grid


ROUTINES = ("optimize_reg", "optimize_constraint_constant")


def _write_record(
    raw_dir: Path,
    *,
    routine: str,
    learning_rate: float,
    lmbda: float,
    r: float,
    retraction: str = "qr",
) -> Path:
    stem = f"{routine}_{retraction}_test_lr{learning_rate:g}_lmbda{lmbda:g}"
    curve_path = raw_dir / "curves" / f"{stem}_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(
        json.dumps(
            {
                "routine": routine,
                "retraction": retraction,
                "config": {"learning_rate": learning_rate},
                "iteration_numbers": 3,
                "loss_history": [3.0, 2.0, 1.0],
                "rmse_history": [0.3, 0.2, 0.1],
            }
        )
        + "\n"
    )
    record = {
        "status": "success",
        "routine": routine,
        "retraction": retraction,
        "config": {"learning_rate": learning_rate},
        "iteration_numbers": 3,
        "rank": 2,
        "r": r,
        "lmbda": lmbda,
        "device": "cpu",
        "dtype": "float64",
        "initialization": "svd_native",
        "curve_file": str(curve_path),
        "final_rmse": 0.1 + learning_rate,
        "final_loss": 1.0 + lmbda,
        "best_rmse": 0.1,
        "best_loss": 1.0,
        "final_gradient_norm": 0.5,
        "final_x_norm": 0.6,
        "run_timestamp": "20260527_120000",
        "config_index": 1,
        "total_configs": 1,
    }
    if routine == "optimize_reg":
        record["final_objective"] = 1.5 + lmbda
    path = raw_dir / f"{stem}.json"
    path.write_text(json.dumps(record) + "\n")
    return path


def _write_grid(raw_dir: Path, *, learning_rates=(0.01, 0.001), lmbdas=(1e-2, 1e-4)) -> None:
    for learning_rate in learning_rates:
        for lmbda in lmbdas:
            for routine in ROUTINES:
                _write_record(raw_dir, routine=routine, learning_rate=learning_rate, lmbda=lmbda, r=10.0 / lmbda)


def test_summarize_constant_step_grid_writes_markdown_table(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_grid(raw_dir)

    output = summarize_constant_step_grid.write_summary(
        raw_dirs=[raw_dir],
        processed_dir=processed_dir,
        retraction="qr",
        expected_records=8,
        timestamp="20260527_120000",
    )

    text = output.read_text()
    assert output.name == "constant_step_optimize_reg_vs_optimize_constraint_constant_qr_summary_20260527_120000.md"
    assert "- Records: 8" in text
    assert "| routine | learning rate | lambda | r | retraction | status |" in text
    assert "optimize_reg" in text
    assert "optimize_constraint_constant" in text


def test_plot_constant_step_grid_writes_one_figure_per_learning_rate(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_grid(raw_dir)

    outputs = plot_constant_step_grid.create_figures(
        raw_dirs=[raw_dir],
        processed_dir=processed_dir,
        retraction="qr",
        timestamp="20260527_120000",
    )

    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)
    assert all("optimize_reg_vs_optimize_constraint_constant_qr" in path.name for path in outputs)


def test_plot_constant_step_grid_rejects_missing_pair(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_record(raw_dir, routine="optimize_reg", learning_rate=0.01, lmbda=1e-2, r=1000.0)

    try:
        plot_constant_step_grid.create_figures(raw_dirs=[raw_dir], processed_dir=tmp_path / "processed", retraction="qr")
    except ValueError as exc:
        assert "Missing record" in str(exc)
    else:
        raise AssertionError("Expected missing pair to fail loudly.")
