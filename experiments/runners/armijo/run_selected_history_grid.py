from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manifold_opt.optim.initial import initial_lr
from experiments.runners.shared.armijo_grid import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    DTYPES,
    RETRACTIONS,
    ROUTINES,
    GridConfig,
    _armijo_evaluation_summary,
    config_id,
    load_masked_dataset,
    load_record,
    parse_csv_floats,
    parse_csv_ints,
    parse_r_value,
    resolve_device,
    resolve_radius,
    run_config,
    write_best_summary,
    write_json_atomic,
    write_reproducibility_record,
)
from experiments.runners.shared.armijo_layout import config_path, grid_config_path, write_grid_config


@dataclass(frozen=True)
class SourceCandidate:
    """A successful final-mode grid record that can seed one history-mode rerun."""

    source_run_id: str
    source_file: str
    source_config_id: str
    routine: str
    retraction: str
    rank: int
    lmbda: float
    r: float
    config: GridConfig
    final_rmse: float
    final_wlra_loss: float
    candidate_rank: int = 0


def _finite_float(record: dict[str, object], key: str, *, default: float = math.inf) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _candidate_from_record(path: Path, record: dict[str, object]) -> SourceCandidate | None:
    if record.get("status") != "success" or record.get("metric_mode") != "final":
        return None
    config = record.get("config")
    if not isinstance(config, dict):
        return None
    routine = str(config.get("routine", ""))
    if routine not in ROUTINES:
        return None
    retraction = str(record.get("retraction", ""))
    if retraction not in RETRACTIONS:
        return None
    try:
        rank = int(record["rank"])
        lmbda = float(record["lmbda"])
        r = float(record["r"])
        beta = float(config["beta"])
        sigma = float(config["sigma"])
        initial_step = float(config["initial_step"])
        max_backtracks = int(config["max_backtracks"])
    except (KeyError, TypeError, ValueError):
        return None
    final_rmse = _finite_float(record, "final_rmse")
    final_wlra_loss = _finite_float(record, "final_wlra_loss")
    if not math.isfinite(final_rmse) or not math.isfinite(final_wlra_loss):
        return None
    grid_config = GridConfig(
        routine=routine,
        beta=beta,
        sigma=sigma,
        initial_step=initial_step,
        max_backtracks=max_backtracks,
    )
    source_config_id = str(
        record.get("config_id") or config_id(grid_config, retraction=retraction, rank=rank, lmbda=lmbda)
    )
    return SourceCandidate(
        source_run_id=path.parents[2].name,
        source_file=str(path),
        source_config_id=source_config_id,
        routine=routine,
        retraction=retraction,
        rank=rank,
        lmbda=lmbda,
        r=r,
        config=grid_config,
        final_rmse=final_rmse,
        final_wlra_loss=final_wlra_loss,
    )


def _source_run_dirs(raw_root: Path, source_run_prefix: str) -> list[Path]:
    return sorted(
        path
        for path in raw_root.glob(f"{source_run_prefix}_*")
        if path.is_dir() and not path.name.endswith("_logs")
    )


def load_source_candidates(raw_root: Path, *, source_run_prefix: str) -> list[SourceCandidate]:
    """Load all successful final-mode records from the source grid prefix."""
    candidates: list[SourceCandidate] = []
    for run_dir in _source_run_dirs(raw_root, source_run_prefix):
        for path in sorted(run_dir.rglob("*.json")):
            if path.name == "grid_config.json":
                continue
            record = load_record(path)
            if not record:
                continue
            candidate = _candidate_from_record(path, record)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def rank_source_candidates(
    candidates: Iterable[SourceCandidate],
    *,
    routine: str,
    retraction: str,
    ranks: Iterable[int],
    lmbdas: Iterable[float],
) -> dict[tuple[str, int, float, str], list[SourceCandidate]]:
    """Rank source candidates for each target group by final RMSE and fair loss."""
    wanted_ranks = set(int(rank) for rank in ranks)
    wanted_lmbdas = [float(value) for value in lmbdas]
    grouped: dict[tuple[str, int, float, str], list[SourceCandidate]] = {
        (retraction, rank, lmbda, routine): [] for rank in wanted_ranks for lmbda in wanted_lmbdas
    }
    for candidate in candidates:
        if candidate.routine != routine or candidate.retraction != retraction:
            continue
        if candidate.rank not in wanted_ranks:
            continue
        matched_lmbda = next(
            (
                value
                for value in wanted_lmbdas
                if math.isclose(candidate.lmbda, value, rel_tol=1e-12, abs_tol=1e-12)
            ),
            None,
        )
        if matched_lmbda is None:
            continue
        grouped[(retraction, candidate.rank, matched_lmbda, routine)].append(candidate)

    ranked: dict[tuple[str, int, float, str], list[SourceCandidate]] = {}
    for group, values in grouped.items():
        ordered = sorted(values, key=lambda item: (item.final_rmse, item.final_wlra_loss, item.source_file))
        ranked[group] = [
            SourceCandidate(
                source_run_id=item.source_run_id,
                source_file=item.source_file,
                source_config_id=item.source_config_id,
                routine=item.routine,
                retraction=item.retraction,
                rank=item.rank,
                lmbda=item.lmbda,
                r=item.r,
                config=item.config,
                final_rmse=item.final_rmse,
                final_wlra_loss=item.final_wlra_loss,
                candidate_rank=index,
            )
            for index, item in enumerate(ordered, start=1)
        ]
    return ranked


def _load_target_records(run_raw_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not run_raw_dir.exists():
        return records
    for path in sorted(run_raw_dir.rglob("*.json")):
        if path.name == "grid_config.json":
            continue
        record = load_record(path)
        if record:
            record["_target_file"] = str(path)
            records.append(record)
    return records


def _record_group(record: dict[str, object]) -> tuple[str, int, float, str] | None:
    config = record.get("config")
    if not isinstance(config, dict):
        return None
    try:
        return (
            str(record["retraction"]),
            int(record["rank"]),
            float(record["lmbda"]),
            str(config["routine"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _matching_success(
    records: Iterable[dict[str, object]],
    group: tuple[str, int, float, str],
    *,
    iteration_numbers: int,
    armijo_tol: float,
    rmse_mode: str,
) -> dict[str, object] | None:
    for record in records:
        record_group = _record_group(record)
        if record_group is None:
            continue
        if (
            record.get("status") == "success"
            and record_group[0] == group[0]
            and record_group[1] == group[1]
            and math.isclose(record_group[2], group[2], rel_tol=1e-12, abs_tol=1e-12)
            and record_group[3] == group[3]
            and record.get("iteration_numbers") == iteration_numbers
            and record.get("metric_mode") == rmse_mode
            and math.isclose(float(record.get("armijo_tol", math.nan)), armijo_tol, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return record
    return None


def _attempted_source_ids(records: Iterable[dict[str, object]], group: tuple[str, int, float, str]) -> set[str]:
    result: set[str] = set()
    for record in records:
        record_group = _record_group(record)
        if record_group is None:
            continue
        if (
            record_group[0] == group[0]
            and record_group[1] == group[1]
            and math.isclose(record_group[2], group[2], rel_tol=1e-12, abs_tol=1e-12)
            and record_group[3] == group[3]
        ):
            source_config_id = record.get("source_config_id")
            if source_config_id:
                result.add(str(source_config_id))
    return result


def _next_config_index(records: Iterable[dict[str, object]], *, routine: str, retraction: str) -> int:
    max_index = 0
    for record in records:
        config = record.get("config")
        if not isinstance(config, dict):
            continue
        if config.get("routine") != routine or record.get("retraction") != retraction:
            continue
        try:
            max_index = max(max_index, int(record.get("config_index", 0)))
        except (TypeError, ValueError):
            continue
    return max_index + 1


def _candidate_metadata(candidate: SourceCandidate) -> dict[str, object]:
    return {
        "source_run_id": candidate.source_run_id,
        "source_file": candidate.source_file,
        "source_config_id": candidate.source_config_id,
        "source_candidate_rank": candidate.candidate_rank,
        "source_final_rmse": candidate.final_rmse,
        "source_final_wlra_loss": candidate.final_wlra_loss,
    }


def write_attempt_manifest(run_raw_dir: Path, manifest_path: Path) -> Path:
    """Write a CSV manifest of selected-history attempts in a processed run folder."""
    records = _load_target_records(run_raw_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "routine",
        "retraction",
        "rank",
        "lmbda",
        "candidate_rank",
        "source_config_id",
        "source_file",
        "source_final_rmse",
        "source_final_wlra_loss",
        "status",
        "selected",
        "final_rmse",
        "final_wlra_loss",
        "error_type",
        "error",
        "target_file",
    ]
    rows = []
    for record in records:
        config = record.get("config")
        if not isinstance(config, dict):
            continue
        rows.append(
            {
                "run_id": record.get("run_id", ""),
                "routine": config.get("routine", ""),
                "retraction": record.get("retraction", ""),
                "rank": record.get("rank", ""),
                "lmbda": record.get("lmbda", ""),
                "candidate_rank": record.get("source_candidate_rank", ""),
                "source_config_id": record.get("source_config_id", ""),
                "source_file": record.get("source_file", ""),
                "source_final_rmse": record.get("source_final_rmse", ""),
                "source_final_wlra_loss": record.get("source_final_wlra_loss", ""),
                "status": record.get("status", ""),
                "selected": record.get("status") == "success",
                "final_rmse": record.get("final_rmse", ""),
                "final_wlra_loss": record.get("final_wlra_loss", ""),
                "error_type": record.get("error_type", ""),
                "error": record.get("error", ""),
                "target_file": record.get("_target_file", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["retraction"]),
            str(row["routine"]),
            int(row["rank"] or 0),
            float(row["lmbda"] or 0.0),
            int(row["candidate_rank"] or 0),
            str(row["target_file"]),
        )
    )
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_processed_outputs(run_raw_dir: Path, run_processed_dir: Path) -> None:
    write_best_summary(run_raw_dir, run_processed_dir / "best_configs.csv")
    write_attempt_manifest(run_raw_dir, run_processed_dir / "selected_history_attempts.csv")


def _write_failure_record(
    *,
    path: Path,
    candidate: SourceCandidate,
    run_id: str,
    config_index: int,
    total_groups: int,
    iteration_numbers: int,
    armijo_tol: float,
    rmse_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    error: Exception,
) -> None:
    failure = {
        "config_id": config_id(candidate.config, retraction=candidate.retraction, rank=candidate.rank, lmbda=candidate.lmbda),
        "status": "failed",
        "config": asdict(candidate.config),
        "run_id": run_id,
        "config_index": config_index,
        "total_groups": total_groups,
        "iteration_numbers": iteration_numbers,
        "rank": candidate.rank,
        "r": candidate.r,
        "lmbda": candidate.lmbda,
        "armijo_tol": armijo_tol,
        "retraction": candidate.retraction,
        "metric_mode": rmse_mode,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": datetime.now().isoformat(timespec="seconds"),
    }
    failure.update(_candidate_metadata(candidate))
    history = getattr(error, "armijo_evaluation_history", None)
    if history is not None:
        failure.update(_armijo_evaluation_summary(history))
    write_json_atomic(path, failure)


def run_selected_history_grid(
    *,
    dataset_path: str | Path,
    source_run_prefix: str,
    run_name: str,
    routine: str,
    retraction: str,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    ranks: Iterable[int] = (32, 64, 128),
    lmbdas: Iterable[float] = (1e-2, 1e-4, 1e-6),
    iteration_numbers: int = 1000,
    r: float | None = None,
    rmse_mode: str = "history",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    armijo_tol: float = 1e-12,
    force: bool = False,
) -> list[dict[str, object]]:
    """Rerun the best available source-grid candidates until each history curve succeeds."""
    if routine not in ROUTINES:
        raise ValueError(f"Unknown routine: {routine}.")
    if retraction not in RETRACTIONS:
        raise ValueError(f"Unknown retraction: {retraction}.")
    if rmse_mode != "history":
        raise ValueError("Selected history grid runs require rmse_mode='history'.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")
    rank_values = [int(rank) for rank in ranks]
    lmbda_values = [float(value) for value in lmbdas]
    if not rank_values or any(rank <= 0 for rank in rank_values):
        raise ValueError("Each rank must be positive.")
    if not lmbda_values or any(value <= 0 or not math.isfinite(value) for value in lmbda_values):
        raise ValueError("Each lambda must be positive and finite.")

    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir)
    figures_root = Path(figures_dir)
    run_raw_dir = raw_root / run_name
    run_processed_dir = processed_root / run_name
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    run_processed_dir.mkdir(parents=True, exist_ok=True)

    source_candidates = load_source_candidates(raw_root, source_run_prefix=source_run_prefix)
    ranked = rank_source_candidates(
        source_candidates,
        routine=routine,
        retraction=retraction,
        ranks=rank_values,
        lmbdas=lmbda_values,
    )
    missing = [group for group, candidates in sorted(ranked.items()) if not candidates]
    if missing:
        raise ValueError(f"No successful source candidates for group(s): {missing}")

    dataset_path = Path(dataset_path)
    a_full, a_masked, w = load_masked_dataset(dataset_path, dtype=dtype, device=device)
    selected_betas = sorted({candidate.config.beta for candidates in ranked.values() for candidate in candidates})
    selected_sigmas = sorted({candidate.config.sigma for candidates in ranked.values() for candidate in candidates})
    selected_steps = sorted({candidate.config.initial_step for candidates in ranked.values() for candidate in candidates})
    selected_backtracks = sorted({candidate.config.max_backtracks for candidates in ranked.values() for candidate in candidates})
    metadata = {
        "run_id": run_name,
        "dataset_path": str(dataset_path),
        "dataset_file": dataset_path.name,
        "source_run_prefix": source_run_prefix,
        "source_ranking": "final_rmse_then_final_wlra_loss",
        "routines": [routine],
        "retractions": [retraction],
        "ranks": rank_values,
        "lmbdas": lmbda_values,
        "betas": selected_betas,
        "sigmas": selected_sigmas,
        "initial_steps": selected_steps,
        "max_backtracks": selected_backtracks,
        "r_policy": "auto_norm_over_sqrt_lmbda" if r is None else "fixed",
        "iteration_numbers": iteration_numbers,
        "armijo_tol": armijo_tol,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "metric_mode": rmse_mode,
        "raw_dir": str(run_raw_dir),
        "processed_dir": str(run_processed_dir),
        "figures_dir": str(figures_root / run_name),
        "total_groups": len(rank_values) * len(lmbda_values),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_grid_config(grid_config_path(run_processed_dir), metadata)
    write_reproducibility_record(run_processed_dir, metadata)

    failures_without_success: list[tuple[str, int, float, str]] = []
    initial_points = {rank: initial_lr(a_masked, rank) for rank in rank_values}

    for rank in rank_values:
        for lmbda in lmbda_values:
            group = (retraction, rank, lmbda, routine)
            records = _load_target_records(run_raw_dir)
            if not force and _matching_success(
                records,
                group,
                iteration_numbers=iteration_numbers,
                armijo_tol=armijo_tol,
                rmse_mode=rmse_mode,
            ):
                continue
            attempted = set() if force else _attempted_source_ids(records, group)
            group_succeeded = False
            for candidate in ranked[group]:
                if candidate.source_config_id in attempted:
                    continue
                records = _load_target_records(run_raw_dir)
                config_index = _next_config_index(records, routine=routine, retraction=retraction)
                output_path = config_path(
                    run_raw_dir,
                    routine=routine,
                    retraction=retraction,
                    index=config_index,
                )
                current_r = resolve_radius(r, a_masked, lmbda)
                candidate = SourceCandidate(
                    source_run_id=candidate.source_run_id,
                    source_file=candidate.source_file,
                    source_config_id=candidate.source_config_id,
                    routine=candidate.routine,
                    retraction=candidate.retraction,
                    rank=candidate.rank,
                    lmbda=candidate.lmbda,
                    r=current_r,
                    config=candidate.config,
                    final_rmse=candidate.final_rmse,
                    final_wlra_loss=candidate.final_wlra_loss,
                    candidate_rank=candidate.candidate_rank,
                )
                try:
                    record = run_config(
                        candidate.config,
                        a_full=a_full,
                        a_masked=a_masked,
                        w=w,
                        initial_point=initial_points[rank],
                        k=rank,
                        iteration_numbers=iteration_numbers,
                        r=current_r,
                        lmbda=lmbda,
                        armijo_tol=armijo_tol,
                        retraction=retraction,
                        rmse_mode=rmse_mode,
                    )
                except Exception as exc:
                    _write_failure_record(
                        path=output_path,
                        candidate=candidate,
                        run_id=run_name,
                        config_index=config_index,
                        total_groups=len(rank_values) * len(lmbda_values),
                        iteration_numbers=iteration_numbers,
                        armijo_tol=armijo_tol,
                        rmse_mode=rmse_mode,
                        device=device,
                        dtype=dtype,
                        error=exc,
                    )
                    _write_processed_outputs(run_raw_dir, run_processed_dir)
                    continue
                record["run_id"] = run_name
                record["config_index"] = config_index
                record["total_groups"] = len(rank_values) * len(lmbda_values)
                record.update(_candidate_metadata(candidate))
                write_json_atomic(output_path, record)
                _write_processed_outputs(run_raw_dir, run_processed_dir)
                group_succeeded = True
                break
            if not group_succeeded:
                failures_without_success.append(group)

    _write_processed_outputs(run_raw_dir, run_processed_dir)
    if failures_without_success:
        raise RuntimeError(f"No successful history records for group(s): {failures_without_success}")
    return write_best_summary(run_raw_dir, run_processed_dir / "best_configs.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run selected 1000-iteration history curves from a completed Armijo grid.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--source-run-prefix", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--routine", choices=ROUTINES, required=True)
    parser.add_argument("--retraction", choices=RETRACTIONS, required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--ranks", type=parse_csv_ints, default=[32, 64, 128])
    parser.add_argument("--lmbdas", type=parse_csv_floats, default=[1e-2, 1e-4, 1e-6])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--r", type=parse_r_value, default=None)
    parser.add_argument("--rmse-mode", choices=("history",), default="history")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--armijo-tol", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "mps" and args.dtype == "float64":
        raise ValueError("MPS does not support float64; use --dtype float32.")
    run_selected_history_grid(
        dataset_path=args.dataset_path,
        source_run_prefix=args.source_run_prefix,
        run_name=args.run_name,
        routine=args.routine,
        retraction=args.retraction,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        figures_dir=args.figures_dir,
        ranks=args.ranks,
        lmbdas=args.lmbdas,
        iteration_numbers=args.iterations,
        r=args.r,
        rmse_mode=args.rmse_mode,
        device=device,
        dtype=DTYPES[args.dtype],
        armijo_tol=args.armijo_tol,
        force=args.force,
    )


if __name__ == "__main__":
    main()
