from __future__ import annotations

import json

import pytest

import experiments.summaries.armijo.summarize_armijo_convergence_methods as summarize_armijo_convergence_methods


def test_qr_convergence_methods_summary_parses_best_median_worst(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    path = processed_dir / "armijo_projection_arc_qr_summary_test.md"
    path.write_text(
        "\n".join(
            [
                "# Projection-Arc QR Grid Search Summary",
                "",
                "- Source: `results/raws/armijo/projection_arc/qr/20260528_012655__lmbda-1e-4__iters-5000__device-cpu/armijo_projection_arc_qr_*.json`",
                "- Records: 4",
                "- Device: cpu",
                "",
                "| beta | sigma | s-bar | r | max-backtracks | final rmse | final loss | approx runtime |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "| 0.5 | 0.25 | 0.1 | 100 | 200 | 0.4 | 10 | 0h 1m 0s |",
                "| 0.9 | 0.75 | 0.3 | 100 | 200 | 0.2 | 8 | 0h 2m 0s |",
                "| 0.75 | 0.5 | 1 | 100 | 200 | 0.3 | 9 | 0h 3m 0s |",
                "| 0.9 | 0.75 | 1 | 100 | 200 |  |  | 0h 4m 0s |",
                "",
            ]
        )
    )

    summary = summarize_armijo_convergence_methods.summarize_file(path)

    assert summary.method == "projection-arc"
    assert summary.records == 4
    assert summary.successful == 3
    assert summary.failed == 1
    assert summary.device == "cpu"
    assert summary.average_runtime_seconds == 150
    assert summary.average_rmse == pytest.approx(0.3)
    assert summary.average_loss == 9.0
    assert summary.dataset_files == ("mnist0_n600_mask0.70_seed42_train.pt",)
    assert summary.lmbdas == (1e-4,)
    assert summary.rs == (100.0,)
    assert summary.max_backtracks == (200.0,)
    assert summary.best["rmse"] == 0.2
    assert summary.best["step_label"] == "s-bar"
    assert summary.median["rmse"] == 0.3
    assert summary.worst["rmse"] == 0.4

    text = summarize_armijo_convergence_methods.markdown_summary([summary])
    assert "- Dataset file: mnist0_n600_mask0.70_seed42_train.pt" in text
    assert "- Lambda: 0.0001" in text
    assert "- r: 100" in text
    assert "- Max backtracks: 200" in text
    assert "| projection-arc | 4 | 3/1 | cpu | 0h 2m 30s | 0.3 | 9 |" in text
    assert "beta=0.9, sigma=0.75, s-bar=0.3" in text


def test_latest_matching_file_uses_newest_summary_regardless_record_count(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    older = processed_dir / "armijo_projection_arc_qr_summary_old.md"
    newer = processed_dir / "armijo_projection_arc_qr_summary_new.md"
    older.write_text("- Records: 27\n")
    newer.write_text("- Records: 45\n")

    older.touch()
    newer.touch()

    assert (
        summarize_armijo_convergence_methods.latest_matching_file(
            processed_dir,
            "armijo_projection_arc_qr_summary_*.md",
        )
        == newer
    )


def test_json_summary_groups_by_retraction_method_rank_and_lambda(tmp_path):
    raw_dir = tmp_path / "raw" / "260609-01"
    processed_dir = tmp_path / "processed" / "260609-01"
    (raw_dir / "regularized" / "qr").mkdir(parents=True)
    (raw_dir / "projection_arc" / "polar").mkdir(parents=True)
    (raw_dir / "feasible_direction" / "qr").mkdir(parents=True)
    (raw_dir / "grid_config.json").write_text(json.dumps({"dataset_file": "mnist_train-full_mask0.50_seed42.pt"}))
    records = [
        (
            raw_dir / "regularized" / "qr" / "regularized_qr_001.json",
            {
                "status": "success",
                "retraction": "qr",
                "rank": 32,
                "lmbda": 1e-4,
                "r": 10.0,
                "config": {"routine": "armijo_wlra_reg", "beta": 0.5, "sigma": 0.25, "initial_step": 0.1},
                "final_rmse": 0.2,
                "final_wlra_loss": 8.0,
                "final_regularized_objective": 99.0,
                "mean_armijo_evaluations": 2.0,
            },
        ),
        (
            raw_dir / "regularized" / "qr" / "regularized_qr_002.json",
            {
                "status": "success",
                "retraction": "qr",
                "rank": 32,
                "lmbda": 1e-4,
                "r": 10.0,
                "config": {"routine": "armijo_wlra_reg", "beta": 0.9, "sigma": 0.5, "initial_step": 1.0},
                "final_rmse": 0.3,
                "final_wlra_loss": 6.0,
                "final_regularized_objective": 7.0,
                "mean_armijo_evaluations": 4.0,
            },
        ),
        (
            raw_dir / "projection_arc" / "polar" / "projection_arc_polar_001.json",
            {
                "status": "success",
                "retraction": "polar",
                "rank": 64,
                "lmbda": 1e-2,
                "r": 20.0,
                "config": {"routine": "armijo_projection_arc", "beta": 0.7, "sigma": 0.25, "initial_step": 0.3},
                "final_rmse": 0.4,
                "final_wlra_loss": 12.0,
                "mean_armijo_evaluations": 3.0,
            },
        ),
        (
            raw_dir / "feasible_direction" / "qr" / "feasible_direction_qr_001.json",
            {
                "status": "failed",
                "retraction": "qr",
                "rank": 32,
                "lmbda": 1e-4,
                "config": {"routine": "armijo_feasible_direction", "beta": 0.5, "sigma": 0.25, "initial_step": 0.1},
                "error_type": "RuntimeError",
                "error": "boom",
            },
        ),
    ]
    for path, record in records:
        path.write_text(json.dumps(record))

    output = summarize_armijo_convergence_methods.write_summary(processed_dir=processed_dir, raw_dir=raw_dir)

    text = output.read_text()
    assert output.name == "convergence_methods_all_summary.md"
    assert "## QR Main Summary" in text
    assert "## POLAR Main Summary" in text
    assert "best final wlra loss" in text
    assert "mean armijo evals" in text
    assert "| qr | regularized-wlra | 32 | 0.0001 | 10 | 0.2 | 8 | 0.5 | 0.25 | 0.1 |" in text
    assert "| 0.25 | 7 | 3 | 2 | 0 |" in text
    assert "final_regularized_objective" not in text
    assert "| qr | feasible-direction | 32 | 0.0001 | 0.5 | 0.25 | 0.1 | failed | RuntimeError | boom |" in text


def test_manuscript_table_groups_retractions_and_routine_columns(tmp_path):
    raw_root = tmp_path / "raw"
    output_tex = tmp_path / "table.tex"

    def write_record(run_name, routine_dir, retraction, index, record):
        run_dir = raw_root / run_name
        record_dir = run_dir / routine_dir / retraction
        record_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "grid_config.json").write_text(json.dumps({"dataset_file": "mnist_train-full_mask0.50_seed42.pt"}))
        path = record_dir / f"{routine_dir}_{retraction}_{index:03d}.json"
        path.write_text(json.dumps(record))

    def success(routine, retraction, rank, lmbda, rmse, loss, beta, sigma, step, mean_evals):
        return {
            "status": "success",
            "retraction": retraction,
            "rank": rank,
            "lmbda": lmbda,
            "config": {"routine": routine, "beta": beta, "sigma": sigma, "initial_step": step},
            "final_rmse": rmse,
            "final_wlra_loss": loss,
            "final_regularized_objective": 999.0,
            "mean_armijo_evaluations": mean_evals,
        }

    def failure(routine, retraction, rank, lmbda, beta, sigma, step):
        return {
            "status": "failed",
            "retraction": retraction,
            "rank": rank,
            "lmbda": lmbda,
            "config": {"routine": routine, "beta": beta, "sigma": sigma, "initial_step": step},
            "error_type": "RuntimeError",
            "error": "boom",
        }

    write_record(
        "full_train_20260618_qr_regularized",
        "regularized",
        "qr",
        1,
        success("armijo_wlra_reg", "qr", 32, 1e-2, 0.2, 9.0, 0.5, 0.25, 0.1, 2.0),
    )
    write_record(
        "full_train_20260618_qr_regularized",
        "regularized",
        "qr",
        2,
        success("armijo_wlra_reg", "qr", 32, 1e-2, 0.2, 7.0, 0.75, 0.5, 0.3, 4.0),
    )
    write_record(
        "full_train_20260618_qr_regularized",
        "regularized",
        "qr",
        3,
        success("armijo_wlra_reg", "qr", 32, 1e-2, 0.4, 6.0, 0.9, 0.5, 1.0, 8.0),
    )
    write_record(
        "full_train_20260618_qr_regularized",
        "regularized",
        "qr",
        4,
        failure("armijo_wlra_reg", "qr", 32, 1e-2, 0.5, 0.75, 0.5),
    )
    write_record(
        "full_train_20260618_qr_feasible_direction",
        "feasible_direction",
        "qr",
        1,
        success("armijo_feasible_direction", "qr", 32, 1e-2, 0.3, 8.0, 0.5, 0.5, 0.5, 5.0),
    )
    write_record(
        "full_train_20260618_qr_projection_arc",
        "projection_arc",
        "qr",
        1,
        success("armijo_projection_arc", "qr", 32, 1e-2, 0.25, 7.0, 0.9, 0.75, 0.3, 7.0),
    )
    write_record(
        "full_train_20260618_qr_projection_arc",
        "projection_arc",
        "qr",
        2,
        success("armijo_projection_arc", "qr", 64, 1e-4, 0.5, 6.0, 0.75, 0.25, 0.5, 9.0),
    )
    for routine, routine_dir in [
        ("armijo_wlra_reg", "regularized"),
        ("armijo_feasible_direction", "feasible_direction"),
        ("armijo_projection_arc", "projection_arc"),
    ]:
        write_record(
            f"full_train_20260618_polar_{routine_dir}",
            routine_dir,
            "polar",
            1,
            success(routine, "polar", 32, 1e-2, 0.35, 10.0, 0.5, 0.25, 0.3, 6.0),
        )

    output = summarize_armijo_convergence_methods.write_manuscript_table(
        raw_root=raw_root,
        run_prefix="full_train_20260618",
        output_tex=output_tex,
    )

    text = output.read_text()
    assert output == output_tex
    assert r"\multicolumn{11}{c}{\textbf{QR retraction}}" in text
    assert r"\multicolumn{11}{c}{\textbf{POLAR retraction}}" in text
    regularized_header = r"\multicolumn{3}{c|}{\shortstack{Regularized Armijo\\(Benchmark)}}"
    feasible_header = r"\multicolumn{3}{c|}{\shortstack{Feasible Direction\\(Main Routine)}}"
    projection_header = r"\multicolumn{3}{c}{\shortstack{Projection Arc\\(Main Routine)}}"
    assert text.index(regularized_header) < text.index(feasible_header)
    assert text.index(feasible_header) < text.index(projection_header)
    assert "worst" not in text
    assert "best RMSE" in text
    assert "mean RMSE" in text
    assert "mean evals" not in text
    assert "failures" in text
    assert text.index(r"32 & $10^{-2}$") < text.index(r"64 & $10^{-4}$")
    assert r"\textbf{0.2000 $(0.75,0.5,0.3)$} & 0.2667 & 1" in text
    assert r"& 0.3000 $(0.5,0.5,0.5)$ & 0.3000 & 0 &" in text
    assert r"\textbf{0.3000 $(0.5,0.5,0.5)$}" not in text
    assert "final_regularized_objective" not in text

    output_with_evals = summarize_armijo_convergence_methods.write_manuscript_table(
        raw_root=raw_root,
        run_prefix="full_train_20260618",
        output_tex=tmp_path / "table_with_evals.tex",
        include_armijo_evals=True,
    )

    text_with_evals = output_with_evals.read_text()
    assert r"\multicolumn{14}{c}{\textbf{QR retraction}}" in text_with_evals
    assert r"\multicolumn{4}{c|}{\shortstack{Regularized Armijo\\(Benchmark)}}" in text_with_evals
    assert r"\multicolumn{4}{c|}{\shortstack{Feasible Direction\\(Main Routine)}}" in text_with_evals
    assert r"\multicolumn{4}{c}{\shortstack{Projection Arc\\(Main Routine)}}" in text_with_evals
    assert "Routine 1" not in text_with_evals
    assert "Routine 2" not in text_with_evals
    assert r"\shortstack{best\\RMSE} & \shortstack{mean\\RMSE} & \shortstack{mean\\evals} & failures" in text_with_evals
    assert r"\textbf{0.2000 $(0.75,0.5,0.3)$} & 0.2667 & 4.7 & 1" in text_with_evals
    assert r"& 0.3000 $(0.5,0.5,0.5)$ & 0.3000 & 5.0 & 0 &" in text_with_evals
    assert r"\textbf{0.3000 $(0.5,0.5,0.5)$}" not in text_with_evals
