from __future__ import annotations

import json

import pytest

from experiments.summaries.armijo.paired_t_tests import (
    DEFAULT_EXPECTED_PAIRS,
    DEFAULT_LONG_RUN_EXPECTED_PAIRS,
    GRID_SHARD_SUFFIXES,
    PairedTestResult,
    expected_grid_shard_dirs,
    expected_long_run_shard_dirs,
    latex_table,
    load_long_run_dataset_values,
    paired_long_run_final_rmse_values,
    paired_metric_values,
    paired_one_sided_t_test,
    run_long_run_paired_t_tests,
    run_paired_t_tests,
)
from experiments.summaries.armijo.summarize_armijo_convergence_methods import (
    ManuscriptGroupSummary,
)


def _summary(routine: str, best_rmse: float, mean_rmse: float, mean_evals: float):
    return ManuscriptGroupSummary(
        retraction="qr",
        rank=32,
        lmbda=1e-2,
        routine=routine,
        best={"final_rmse": best_rmse},
        mean_rmse=mean_rmse,
        mean_armijo_evaluations=mean_evals,
        failures=0,
    )


def test_paired_metric_values_match_dataset_configurations():
    benchmark = _summary("armijo_wlra_reg", 0.4, 0.5, 8.0)
    routine = _summary("armijo_feasible_direction", 0.3, 0.4, 7.0)
    summaries = {
        ("qr", 32, 1e-2, benchmark.routine): benchmark,
        ("qr", 32, 1e-2, routine.routine): routine,
    }

    benchmark_values, routine_values = paired_metric_values(
        [("dataset", summaries)],
        routine="armijo_feasible_direction",
        metric="best_rmse",
    )

    assert benchmark_values == [0.4]
    assert routine_values == [0.3]


def test_paired_one_sided_t_test_detects_lower_routine_values():
    result = paired_one_sided_t_test(
        [1.0, 1.4, 1.8, 2.2, 2.6, 3.0],
        [0.8, 1.1, 1.7, 1.8, 2.5, 2.7],
        comparison="Feasible Direction (Main Routine) vs Regularized Armijo (Benchmark)",
        metric="best RMSE",
        alpha=0.05,
    )

    assert result.n == 6
    assert result.degrees_of_freedom == 5
    assert result.mean_difference < 0
    assert result.p_value < 0.05
    assert result.outperforms_benchmark is True


def test_paired_one_sided_t_test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        paired_one_sided_t_test(
            [1.0, 2.0],
            [1.0],
            comparison="Feasible Direction (Main Routine) vs Regularized Armijo (Benchmark)",
            metric="best RMSE",
            alpha=0.05,
        )


def test_expected_grid_shard_dirs_excludes_similarly_prefixed_history(tmp_path):
    run_prefix = "dataset_mask025"
    expected = []
    for retraction, routine in GRID_SHARD_SUFFIXES:
        shard_dir = tmp_path / f"{run_prefix}_{retraction}_{routine}"
        shard_dir.mkdir()
        (shard_dir / "grid_config.json").write_text("{}")
        expected.append(shard_dir)

    history_dir = tmp_path / f"{run_prefix}_history1000_qr_regularized"
    history_dir.mkdir()
    (history_dir / "grid_config.json").write_text("{}")

    shard_dirs = expected_grid_shard_dirs(
        raw_root=tmp_path, run_prefix=run_prefix
    )

    assert shard_dirs == tuple(expected)
    assert history_dir not in shard_dirs


def test_expected_grid_shard_dirs_rejects_missing_shard(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing expected raw grid shard"):
        expected_grid_shard_dirs(raw_root=tmp_path, run_prefix="missing")


def test_expected_long_run_shard_dirs_excludes_similarly_prefixed_directory(
    tmp_path,
):
    run_prefix = "dataset_history1000"
    expected = []
    for retraction, routine in GRID_SHARD_SUFFIXES:
        shard_dir = tmp_path / f"{run_prefix}_{retraction}_{routine}"
        shard_dir.mkdir()
        expected.append(shard_dir)

    unrelated = tmp_path / f"{run_prefix}_extra_qr_regularized"
    unrelated.mkdir()

    shard_dirs = expected_long_run_shard_dirs(
        raw_root=tmp_path, run_prefix=run_prefix
    )

    assert shard_dirs == tuple(expected)
    assert unrelated not in shard_dirs


def test_load_long_run_dataset_values_validates_histories_and_counts_failures(
    tmp_path,
):
    run_prefix = "dataset_history1000"
    routine_by_suffix = {
        "regularized": "armijo_wlra_reg",
        "feasible_direction": "armijo_feasible_direction",
        "projection_arc": "armijo_projection_arc",
    }
    expected_values = {}
    for index, (retraction, routine_suffix) in enumerate(GRID_SHARD_SUFFIXES):
        shard_dir = tmp_path / f"{run_prefix}_{retraction}_{routine_suffix}"
        shard_dir.mkdir()
        routine = routine_by_suffix[routine_suffix]
        final_rmse = 0.2 + 0.01 * index
        record = {
            "status": "success",
            "config": {"routine": routine},
            "retraction": retraction,
            "rank": 32,
            "lmbda": 1e-2,
            "iteration_numbers": 3,
            "rmse_history": [final_rmse + 0.02, final_rmse + 0.01, final_rmse],
            "final_rmse": final_rmse,
        }
        (shard_dir / "success.json").write_text(json.dumps(record))
        expected_values[(retraction, 32, 1e-2, routine)] = final_rmse

    failed_record = {
        "status": "failed",
        "config": {"routine": "armijo_projection_arc"},
        "retraction": "polar",
        "rank": 32,
        "lmbda": 1e-2,
    }
    failed_shard = (
        tmp_path / f"{run_prefix}_polar_projection_arc" / "failed.json"
    )
    failed_shard.write_text(json.dumps(failed_record))

    loaded, failed_attempts = load_long_run_dataset_values(
        raw_root=tmp_path,
        dataset_prefixes=[("dataset", run_prefix)],
        expected_history_length=3,
        expected_ranks=(32,),
        expected_lmbdas=(1e-2,),
    )

    assert loaded == [("dataset", expected_values)]
    assert failed_attempts == 1


def test_paired_long_run_final_rmse_values_match_configurations():
    values = {
        ("qr", 32, 1e-2, "armijo_wlra_reg"): 0.4,
        ("qr", 32, 1e-2, "armijo_feasible_direction"): 0.3,
    }

    benchmark_values, routine_values = paired_long_run_final_rmse_values(
        [("dataset", values)],
        routine="armijo_feasible_direction",
    )

    assert benchmark_values == [0.4]
    assert routine_values == [0.3]


def test_run_paired_t_tests_uses_126_pairs_and_rmse_metrics_only():
    dataset_summaries = []
    for index in range(DEFAULT_EXPECTED_PAIRS):
        benchmark_best = 1.0 + 0.01 * index
        benchmark_mean = 1.1 + 0.01 * index
        feasible = _summary(
            "armijo_feasible_direction",
            benchmark_best - 0.05 - 0.001 * (index % 5),
            benchmark_mean - 0.04 - 0.001 * (index % 7),
            7.0,
        )
        projection = _summary(
            "armijo_projection_arc",
            benchmark_best - 0.03 - 0.001 * (index % 7),
            benchmark_mean - 0.02 - 0.001 * (index % 5),
            9.0,
        )
        benchmark = _summary(
            "armijo_wlra_reg", benchmark_best, benchmark_mean, 8.0
        )
        summaries = {
            ("qr", 32, 1e-2, benchmark.routine): benchmark,
            ("qr", 32, 1e-2, feasible.routine): feasible,
            ("qr", 32, 1e-2, projection.routine): projection,
        }
        dataset_summaries.append((f"dataset-{index}", summaries))

    results = run_paired_t_tests(dataset_summaries)

    assert len(results) == 4
    assert [result.metric for result in results] == [
        "best RMSE",
        "mean RMSE",
        "best RMSE",
        "mean RMSE",
    ]
    assert all(result.n == 126 for result in results)
    assert all(result.degrees_of_freedom == 125 for result in results)
    assert all(result.outperforms_benchmark for result in results)


def test_run_long_run_paired_t_tests_uses_108_final_rmse_pairs():
    dataset_values = []
    for index in range(DEFAULT_LONG_RUN_EXPECTED_PAIRS):
        benchmark = 1.0 + 0.01 * index
        values = {
            ("qr", 32, 1e-2, "armijo_wlra_reg"): benchmark,
            (
                "qr",
                32,
                1e-2,
                "armijo_feasible_direction",
            ): benchmark - 0.05 - 0.001 * (index % 5),
            (
                "qr",
                32,
                1e-2,
                "armijo_projection_arc",
            ): benchmark - 0.03 - 0.001 * (index % 7),
        }
        dataset_values.append((f"dataset-{index}", values))

    results = run_long_run_paired_t_tests(dataset_values)

    assert len(results) == 2
    assert [result.metric for result in results] == [
        "final RMSE",
        "final RMSE",
    ]
    assert all(result.n == 108 for result in results)
    assert all(result.degrees_of_freedom == 107 for result in results)
    assert all(result.outperforms_benchmark for result in results)


def test_latex_table_reports_directional_decision():
    result = PairedTestResult(
        comparison="Feasible Direction (Main Routine) vs Regularized Armijo (Benchmark)",
        metric="mean RMSE",
        n=126,
        degrees_of_freedom=125,
        mean_difference=-1e-4,
        t_statistic=-2.5,
        p_value=0.008,
        alpha=0.05,
        outperforms_benchmark=True,
    )

    text = latex_table([result])

    assert r"$t_{125}$" in text
    assert r"$-1.000\times 10^{-4}$" in text
    assert r"\textbf{Yes}" in text
    assert (
        r"\shortstack{Feasible Direction (Main Routine) vs\\Regularized Armijo (Benchmark)}"
        in text
    )
    assert "Routine 1" not in text

    grouped_text = latex_table([result, result])
    assert grouped_text.count(
        r"\shortstack{Feasible Direction (Main Routine) vs\\Regularized Armijo (Benchmark)}"
    ) == 1
