from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.summaries.armijo.summarize_armijo_convergence_methods import (
    FEASIBLE_DIRECTION_DISPLAY,
    ManuscriptGroupSummary,
    PROJECTION_ARC_DISPLAY,
    REGULARIZED_ARMIJO_DISPLAY,
    load_json_records,
    manuscript_group_summaries,
)


DEFAULT_RAW_ROOT = REPO_ROOT / "results" / "raws" / "armijo"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "results"
    / "processed"
    / "armijo"
    / "all_grid_search_paired_t_tests.csv"
)
DEFAULT_OUTPUT_TEX = (
    DEFAULT_OUTPUT_CSV.with_suffix(".tex")
)
DEFAULT_LONG_RUN_OUTPUT_CSV = (
    REPO_ROOT
    / "results"
    / "processed"
    / "armijo"
    / "all_long_run_final_rmse_paired_t_tests.csv"
)
DEFAULT_LONG_RUN_OUTPUT_TEX = (
    DEFAULT_LONG_RUN_OUTPUT_CSV.with_suffix(".tex")
)
DEFAULT_DATASET_PREFIXES = (
    ("Sampled MNIST (masking rate 0.5)", "digits6000_20260707"),
    ("Sampled CIFAR-10 (masking rate 0.5)", "cifar10gray5000_20260708"),
    ("Sampled MNIST (masking rate 0.75)", "digits6000_mask075_20260712"),
    (
        "Sampled CIFAR-10 (masking rate 0.75)",
        "cifar10gray5000_mask075_20260712",
    ),
    ("Sampled MNIST (masking rate 0.25)", "digits6000_mask025_20260718"),
    (
        "Sampled CIFAR-10 (masking rate 0.25)",
        "cifar10gray5000_mask025_20260718",
    ),
    ("Full-training MNIST (masking rate 0.5)", "full_train_20260618"),
)
DEFAULT_LONG_RUN_DATASET_PREFIXES = (
    (
        "Sampled MNIST (masking rate 0.5)",
        "digits6000_20260710_history1000",
    ),
    (
        "Sampled CIFAR-10 (masking rate 0.5)",
        "cifar10gray5000_20260710_history1000",
    ),
    (
        "Sampled MNIST (masking rate 0.75)",
        "digits6000_mask075_20260716_history1000",
    ),
    (
        "Sampled CIFAR-10 (masking rate 0.75)",
        "cifar10gray5000_mask075_20260716_history1000",
    ),
    (
        "Sampled MNIST (masking rate 0.25)",
        "digits6000_mask025_20260718_history1000",
    ),
    (
        "Sampled CIFAR-10 (masking rate 0.25)",
        "cifar10gray5000_mask025_20260718_history1000",
    ),
)
GRID_SHARD_SUFFIXES = tuple(
    (retraction, routine)
    for retraction in ("qr", "polar")
    for routine in ("regularized", "projection_arc", "feasible_direction")
)
DEFAULT_EXPECTED_RECORDS_PER_DATASET = 1458
DEFAULT_EXPECTED_PAIRS = 126
DEFAULT_LONG_RUN_EXPECTED_PAIRS = 108
DEFAULT_LONG_RUN_HISTORY_LENGTH = 1000
DEFAULT_LONG_RUN_RANKS = (32, 64, 128)
DEFAULT_LONG_RUN_LMBDAS = (1e-2, 1e-4, 1e-6)
BENCHMARK_ROUTINE = "armijo_wlra_reg"
ROUTINES = (
    ("armijo_feasible_direction", FEASIBLE_DIRECTION_DISPLAY),
    ("armijo_projection_arc", PROJECTION_ARC_DISPLAY),
)
METRICS = (
    ("best_rmse", "best RMSE"),
    ("mean_rmse", "mean RMSE"),
)
ROUTINE_BY_SHARD_SUFFIX = {
    "regularized": BENCHMARK_ROUTINE,
    "feasible_direction": "armijo_feasible_direction",
    "projection_arc": "armijo_projection_arc",
}
RETRACTION_ORDER = {"qr": 0, "polar": 1}


@dataclass(frozen=True)
class PairedTestResult:
    comparison: str
    metric: str
    n: int
    degrees_of_freedom: int
    mean_difference: float
    t_statistic: float
    p_value: float
    alpha: float
    outperforms_benchmark: bool


def _summary_metric(summary: ManuscriptGroupSummary, metric: str) -> float:
    if metric == "best_rmse":
        value = None if summary.best is None else summary.best.get("final_rmse")
    elif metric == "mean_rmse":
        value = summary.mean_rmse
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            f"Missing finite {metric} for {summary.retraction}, rank={summary.rank}, "
            f"lambda={summary.lmbda}, routine={summary.routine}."
        )
    return float(value)


def expected_grid_shard_dirs(*, raw_root: Path, run_prefix: str) -> tuple[Path, ...]:
    """Return the six exact grid-shard directories for a manuscript dataset."""
    shard_dirs = tuple(
        raw_root / f"{run_prefix}_{retraction}_{routine}"
        for retraction, routine in GRID_SHARD_SUFFIXES
    )
    missing = [
        path
        for path in shard_dirs
        if not path.is_dir() or not (path / "grid_config.json").is_file()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing expected raw grid shard(s) for {run_prefix!r}: {missing_text}."
        )
    return shard_dirs


def load_dataset_summaries(
    *,
    raw_root: Path,
    dataset_prefixes: list[tuple[str, str]],
    expected_records_per_dataset: int = DEFAULT_EXPECTED_RECORDS_PER_DATASET,
) -> list[
    tuple[
        str,
        dict[tuple[str, int, float, str], ManuscriptGroupSummary],
    ]
]:
    """Load summaries from the exact six grid shards for each dataset prefix."""
    loaded = []
    for label, prefix in dataset_prefixes:
        records = []
        for shard_dir in expected_grid_shard_dirs(
            raw_root=raw_root, run_prefix=prefix
        ):
            records.extend(load_json_records(shard_dir))
        if len(records) != expected_records_per_dataset:
            raise ValueError(
                f"Expected {expected_records_per_dataset} raw grid records for "
                f"{label} ({prefix}), found {len(records)}."
            )
        loaded.append((label, manuscript_group_summaries(records)))
    return loaded


def expected_long_run_shard_dirs(
    *, raw_root: Path, run_prefix: str
) -> tuple[Path, ...]:
    """Return the six exact 1000-iteration shard directories for a dataset."""
    shard_dirs = tuple(
        raw_root / f"{run_prefix}_{retraction}_{routine}"
        for retraction, routine in GRID_SHARD_SUFFIXES
    )
    missing = [path for path in shard_dirs if not path.is_dir()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing expected raw long-run shard(s) for {run_prefix!r}: "
            f"{missing_text}."
        )
    return shard_dirs


def _raw_record_routine(record: dict) -> str:
    config = record.get("config")
    if isinstance(config, dict) and config.get("routine") is not None:
        return str(config["routine"])
    return str(record.get("routine", ""))


def load_long_run_dataset_values(
    *,
    raw_root: Path,
    dataset_prefixes: list[tuple[str, str]],
    expected_history_length: int = DEFAULT_LONG_RUN_HISTORY_LENGTH,
    expected_ranks: tuple[int, ...] = DEFAULT_LONG_RUN_RANKS,
    expected_lmbdas: tuple[float, ...] = DEFAULT_LONG_RUN_LMBDAS,
) -> tuple[
    list[tuple[str, dict[tuple[str, int, float, str], float]]],
    int,
]:
    """Load selected final RMSE values from exact long-run shards.

    Failed fallback attempts remain part of the audit count but are excluded from
    the paired values. Each shard must contain one successful 1000-iteration
    history for every expected rank-lambda configuration.
    """
    expected_configurations = {
        (rank, lmbda) for rank in expected_ranks for lmbda in expected_lmbdas
    }
    loaded = []
    total_failures = 0
    for dataset_label, prefix in dataset_prefixes:
        values: dict[tuple[str, int, float, str], float] = {}
        shard_dirs = expected_long_run_shard_dirs(
            raw_root=raw_root, run_prefix=prefix
        )
        for (retraction, routine_suffix), shard_dir in zip(
            GRID_SHARD_SUFFIXES, shard_dirs
        ):
            records = load_json_records(shard_dir)
            unexpected_statuses = {
                str(record.get("status"))
                for record in records
                if record.get("status") not in {"success", "failed"}
            }
            if unexpected_statuses:
                raise ValueError(
                    f"Unexpected record status(es) in {shard_dir}: "
                    f"{sorted(unexpected_statuses)}."
                )
            successful = [
                record for record in records if record.get("status") == "success"
            ]
            failed = [
                record for record in records if record.get("status") == "failed"
            ]
            total_failures += len(failed)
            routine = ROUTINE_BY_SHARD_SUFFIX[routine_suffix]
            successful_configurations = {
                (int(record["rank"]), float(record["lmbda"]))
                for record in successful
            }
            if (
                len(successful) != len(expected_configurations)
                or successful_configurations != expected_configurations
            ):
                raise ValueError(
                    f"Expected one successful long-run record for each rank-lambda "
                    f"configuration in {shard_dir}; found {len(successful)} records "
                    f"covering {sorted(successful_configurations)}."
                )
            for record in successful:
                if record.get("retraction") != retraction:
                    raise ValueError(
                        f"Retraction mismatch in {record.get('_source_file')}."
                    )
                if _raw_record_routine(record) != routine:
                    raise ValueError(
                        f"Routine mismatch in {record.get('_source_file')}."
                    )
                if int(record.get("iteration_numbers", -1)) != expected_history_length:
                    raise ValueError(
                        f"Expected {expected_history_length} iterations in "
                        f"{record.get('_source_file')}."
                    )
                history = record.get("rmse_history")
                if not isinstance(history, list) or len(history) != expected_history_length:
                    raise ValueError(
                        f"Expected RMSE history length {expected_history_length} in "
                        f"{record.get('_source_file')}."
                    )
                final_rmse = record.get("final_rmse")
                if not isinstance(final_rmse, (int, float)) or not math.isfinite(
                    float(final_rmse)
                ):
                    raise ValueError(
                        f"Missing finite final RMSE in {record.get('_source_file')}."
                    )
                if not math.isclose(
                    float(history[-1]),
                    float(final_rmse),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"RMSE history endpoint mismatch in "
                        f"{record.get('_source_file')}."
                    )
                key = (
                    retraction,
                    int(record["rank"]),
                    float(record["lmbda"]),
                    routine,
                )
                if key in values:
                    raise ValueError(
                        f"Duplicate successful long-run configuration for "
                        f"{dataset_label}: {key}."
                    )
                values[key] = float(final_rmse)
        loaded.append((dataset_label, values))
    return loaded, total_failures


def paired_metric_values(
    dataset_summaries: list[
        tuple[
            str,
            dict[tuple[str, int, float, str], ManuscriptGroupSummary],
        ]
    ],
    *,
    routine: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Return Benchmark and routine values paired by dataset and configuration."""
    benchmark_values = []
    routine_values = []
    for dataset_label, summaries in dataset_summaries:
        configurations = {
            (retraction, rank, lmbda)
            for retraction, rank, lmbda, item_routine in summaries
            if item_routine == BENCHMARK_ROUTINE
        }
        ordered_configurations = sorted(
            configurations,
            key=lambda item: (
                RETRACTION_ORDER.get(item[0], 99),
                item[1],
                -item[2],
            ),
        )
        for retraction, rank, lmbda in ordered_configurations:
            benchmark = summaries.get(
                (retraction, rank, lmbda, BENCHMARK_ROUTINE)
            )
            candidate = summaries.get((retraction, rank, lmbda, routine))
            if benchmark is None or candidate is None:
                raise ValueError(
                    f"Incomplete pair for {dataset_label}, {retraction}, rank={rank}, "
                    f"lambda={lmbda}, routine={routine}."
                )
            benchmark_values.append(_summary_metric(benchmark, metric))
            routine_values.append(_summary_metric(candidate, metric))
    return benchmark_values, routine_values


def paired_long_run_final_rmse_values(
    dataset_values: list[
        tuple[str, dict[tuple[str, int, float, str], float]]
    ],
    *,
    routine: str,
) -> tuple[list[float], list[float]]:
    """Return final RMSE values paired by long-run dataset and configuration."""
    benchmark_values = []
    routine_values = []
    for dataset_label, values in dataset_values:
        configurations = {
            (retraction, rank, lmbda)
            for retraction, rank, lmbda, item_routine in values
            if item_routine == BENCHMARK_ROUTINE
        }
        ordered_configurations = sorted(
            configurations,
            key=lambda item: (
                RETRACTION_ORDER.get(item[0], 99),
                item[1],
                -item[2],
            ),
        )
        for retraction, rank, lmbda in ordered_configurations:
            benchmark = values.get(
                (retraction, rank, lmbda, BENCHMARK_ROUTINE)
            )
            candidate = values.get((retraction, rank, lmbda, routine))
            if benchmark is None or candidate is None:
                raise ValueError(
                    f"Incomplete long-run pair for {dataset_label}, {retraction}, "
                    f"rank={rank}, lambda={lmbda}, routine={routine}."
                )
            benchmark_values.append(benchmark)
            routine_values.append(candidate)
    return benchmark_values, routine_values


def paired_one_sided_t_test(
    benchmark_values: list[float],
    routine_values: list[float],
    *,
    comparison: str,
    metric: str,
    alpha: float,
) -> PairedTestResult:
    """Test whether a routine has a lower paired mean metric than Benchmark."""
    benchmark = np.asarray(benchmark_values, dtype=float)
    routine = np.asarray(routine_values, dtype=float)
    if benchmark.ndim != 1 or routine.ndim != 1 or benchmark.shape != routine.shape:
        raise ValueError("Paired samples must be one-dimensional and have equal length.")
    if benchmark.size < 2:
        raise ValueError("At least two paired observations are required.")
    if not np.isfinite(benchmark).all() or not np.isfinite(routine).all():
        raise ValueError("Paired samples must contain only finite values.")

    test = stats.ttest_rel(routine, benchmark, alternative="less")
    p_value = float(test.pvalue)
    if not math.isfinite(p_value):
        raise ValueError(f"Paired t-test returned a non-finite p-value for {comparison}.")
    mean_difference = float(np.mean(routine - benchmark))
    return PairedTestResult(
        comparison=comparison,
        metric=metric,
        n=int(benchmark.size),
        degrees_of_freedom=int(benchmark.size - 1),
        mean_difference=mean_difference,
        t_statistic=float(test.statistic),
        p_value=p_value,
        alpha=alpha,
        outperforms_benchmark=mean_difference < 0 and p_value < alpha,
    )


def run_paired_t_tests(
    dataset_summaries: list[
        tuple[
            str,
            dict[tuple[str, int, float, str], ManuscriptGroupSummary],
        ]
    ],
    *,
    alpha: float = 0.05,
    expected_pairs: int = DEFAULT_EXPECTED_PAIRS,
) -> list[PairedTestResult]:
    """Run all Routine-versus-Benchmark tests used by the manuscript table."""
    results = []
    for routine, comparison in ROUTINES:
        for metric, metric_label in METRICS:
            benchmark_values, routine_values = paired_metric_values(
                dataset_summaries,
                routine=routine,
                metric=metric,
            )
            if len(benchmark_values) != expected_pairs:
                raise ValueError(
                    f"Expected {expected_pairs} paired {metric_label} entries for "
                    f"{comparison}, found {len(benchmark_values)}."
                )
            results.append(
                paired_one_sided_t_test(
                    benchmark_values,
                    routine_values,
                    comparison=f"{comparison} vs {REGULARIZED_ARMIJO_DISPLAY}",
                    metric=metric_label,
                    alpha=alpha,
                )
            )
    return results


def run_long_run_paired_t_tests(
    dataset_values: list[
        tuple[str, dict[tuple[str, int, float, str], float]]
    ],
    *,
    alpha: float = 0.05,
    expected_pairs: int = DEFAULT_LONG_RUN_EXPECTED_PAIRS,
) -> list[PairedTestResult]:
    """Run Routine-versus-Benchmark tests on final long-run RMSE."""
    results = []
    for routine, comparison in ROUTINES:
        benchmark_values, routine_values = paired_long_run_final_rmse_values(
            dataset_values,
            routine=routine,
        )
        if len(benchmark_values) != expected_pairs:
            raise ValueError(
                f"Expected {expected_pairs} paired final RMSE entries for "
                f"{comparison}, found {len(benchmark_values)}."
            )
        results.append(
            paired_one_sided_t_test(
                benchmark_values,
                routine_values,
                comparison=f"{comparison} vs {REGULARIZED_ARMIJO_DISPLAY}",
                metric="final RMSE",
                alpha=alpha,
            )
        )
    return results


def _latex_number(value: float, *, digits: int = 3) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if exponent <= -3 or exponent >= 4:
        coefficient = value / (10**exponent)
        return rf"${coefficient:.{digits - 1}f}\times 10^{{{exponent}}}$"
    return f"{value:.{digits}f}"


def _latex_comparison(comparison: str) -> str:
    """Wrap a routine-versus-benchmark label for the manuscript table."""
    routine, separator, benchmark = comparison.partition(" vs ")
    if not separator:
        return comparison
    return rf"\shortstack{{{routine} vs\\{benchmark}}}"


def latex_table(results: list[PairedTestResult]) -> str:
    """Render the compact paired-test table fragment used in the manuscript."""
    if not results:
        raise ValueError("At least one paired-test result is required.")
    degrees_of_freedom = {result.degrees_of_freedom for result in results}
    if len(degrees_of_freedom) != 1:
        raise ValueError("All manuscript paired tests must use the same degrees of freedom.")
    df = next(iter(degrees_of_freedom))
    lines = [
        r"\begin{tabular}{llrrrc}",
        r"\hline",
        rf"Comparison & Metric & mean difference & $t_{{{df}}}$ & one-sided $p$ & Outperforms? \\",
        r"\hline",
    ]
    previous_comparison = None
    for result in results:
        if previous_comparison is not None and result.comparison != previous_comparison:
            lines.append(r"\hline")
        comparison_cell = (
            ""
            if result.comparison == previous_comparison
            else _latex_comparison(result.comparison)
        )
        decision = r"\textbf{Yes}" if result.outperforms_benchmark else "No"
        lines.append(
            " & ".join(
                [
                    comparison_cell,
                    result.metric,
                    _latex_number(result.mean_difference, digits=4),
                    _latex_number(result.t_statistic, digits=4),
                    _latex_number(result.p_value, digits=3),
                    decision,
                ]
            )
            + r" \\"
        )
        previous_comparison = result.comparison
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_results(
    results: list[PairedTestResult],
    *,
    output_csv: Path,
    output_tex: Path,
) -> tuple[Path, Path]:
    """Write machine-readable and manuscript-ready paired-test results."""
    if not results:
        raise ValueError("At least one paired-test result is required.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(results[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(latex_table(results))
    return output_csv, output_tex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired one-sided t-tests for manuscript Armijo table metrics."
    )
    parser.add_argument(
        "--analysis",
        choices=("grid-search", "long-run"),
        default="grid-search",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--dataset-prefix",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_PREFIX"),
        default=None,
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--expected-records-per-dataset",
        type=int,
        default=DEFAULT_EXPECTED_RECORDS_PER_DATASET,
    )
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument(
        "--expected-history-length",
        type=int,
        default=DEFAULT_LONG_RUN_HISTORY_LENGTH,
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-tex", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis == "grid-search":
        dataset_prefixes = args.dataset_prefix or list(DEFAULT_DATASET_PREFIXES)
        dataset_summaries = load_dataset_summaries(
            raw_root=args.raw_root,
            dataset_prefixes=dataset_prefixes,
            expected_records_per_dataset=args.expected_records_per_dataset,
        )
        results = run_paired_t_tests(
            dataset_summaries,
            alpha=args.alpha,
            expected_pairs=(
                DEFAULT_EXPECTED_PAIRS
                if args.expected_pairs is None
                else args.expected_pairs
            ),
        )
        output_csv = args.output_csv or DEFAULT_OUTPUT_CSV
        output_tex = args.output_tex or DEFAULT_OUTPUT_TEX
    else:
        dataset_prefixes = args.dataset_prefix or list(
            DEFAULT_LONG_RUN_DATASET_PREFIXES
        )
        dataset_values, failed_attempts = load_long_run_dataset_values(
            raw_root=args.raw_root,
            dataset_prefixes=dataset_prefixes,
            expected_history_length=args.expected_history_length,
        )
        results = run_long_run_paired_t_tests(
            dataset_values,
            alpha=args.alpha,
            expected_pairs=(
                DEFAULT_LONG_RUN_EXPECTED_PAIRS
                if args.expected_pairs is None
                else args.expected_pairs
            ),
        )
        output_csv = args.output_csv or DEFAULT_LONG_RUN_OUTPUT_CSV
        output_tex = args.output_tex or DEFAULT_LONG_RUN_OUTPUT_TEX
        print(
            f"Loaded {len(dataset_prefixes)} long-run dataset groups; "
            f"excluded {failed_attempts} failed fallback attempts."
        )
    write_results(results, output_csv=output_csv, output_tex=output_tex)


if __name__ == "__main__":
    main()
