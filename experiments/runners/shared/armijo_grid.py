from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manifold_opt.line_search.armijo import armijo_wlra_reg
from manifold_opt.objectives.losses import wlra_loss
from manifold_opt.optim.gradient_descent import optimize_constraint_armijo
from manifold_opt.optim.initial import initial_lr
from experiments.runners.shared.armijo_layout import (
    allocate_run_id,
    config_path,
    grid_config_path,
    write_grid_config,
)


DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "mnist0_n600_mask0.70_seed42_train.pt"
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "armijo"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "armijo"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures" / "armijo"
ROUTINES = ("armijo_wlra_reg", "armijo_projection_arc", "armijo_feasible_direction")
DTYPES = {"float32": torch.float32, "float64": torch.float64}
RMSE_MODES = ("final", "history")
RETRACTIONS = ("qr", "polar")

Point = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class GridConfig:
    routine: str
    beta: float
    sigma: float
    initial_step: float
    max_backtracks: int


def parse_csv_floats(value: str) -> list[float]:
    """Parse a comma-separated list of finite float values."""
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    if not all(math.isfinite(item) for item in values):
        raise ValueError("Grid values must be finite.")
    return values


def parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated list of non-negative integer values."""
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(item < 0 for item in values):
        raise ValueError("max_backtracks values must be non-negative.")
    return values


def parse_r_value(value: str) -> float | None:
    """Parse a non-negative radius value, or return None for the default formula."""
    if value.strip().lower() == "auto":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("r must be finite or 'auto'.")
    if result < 0.0:
        raise ValueError("r must be non-negative.")
    return result


def parse_csv_routines(value: str) -> list[str]:
    """Parse a comma-separated routine list and reject unknown names."""
    routines = [part.strip() for part in value.split(",") if part.strip()]
    if not routines:
        raise ValueError("Expected at least one routine.")
    if len(routines) == 1 and routines[0].lower() == "all":
        return list(ROUTINES)
    unknown = sorted(set(routines) - set(ROUTINES))
    if unknown:
        raise ValueError(f"Unknown routine(s): {', '.join(unknown)}.")
    return routines


def parse_csv_retractions(value: str) -> list[str]:
    """Parse a comma-separated retraction list and reject unknown names."""
    retractions = [part.strip() for part in value.split(",") if part.strip()]
    if not retractions:
        raise ValueError("Expected at least one retraction.")
    if len(retractions) == 1 and retractions[0].lower() == "all":
        return list(RETRACTIONS)
    unknown = sorted(set(retractions) - set(RETRACTIONS))
    if unknown:
        raise ValueError(f"Unknown retraction(s): {', '.join(unknown)}.")
    return retractions


def resolve_device(device: str) -> torch.device:
    """Resolve CPU/MPS device for a grid-search run."""
    if device == "auto":
        return torch.device("cpu")
    target = torch.device(device)
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
    if target.type not in {"cpu", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', or 'mps'.")
    return target


def config_id(
    config: GridConfig,
    *,
    retraction: str | None = None,
    rank: int | None = None,
    lmbda: float | None = None,
) -> str:
    """Return a deterministic identifier for a grid configuration and optional context."""
    payload_data: dict[str, object] = {"config": asdict(config)}
    if retraction is not None:
        payload_data["retraction"] = retraction
    if rank is not None:
        payload_data["rank"] = rank
    if lmbda is not None:
        payload_data["lmbda"] = lmbda
    payload = json.dumps(payload_data, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{config.routine}_{digest}"


def build_grid(
    *,
    routines: Iterable[str],
    betas: Iterable[float],
    sigmas: Iterable[float],
    initial_steps: Iterable[float],
    max_backtracks_values: Iterable[int],
) -> list[GridConfig]:
    """Build the Cartesian product of Armijo grid-search configurations."""
    routines = list(routines)
    betas = list(betas)
    sigmas = list(sigmas)
    initial_steps = list(initial_steps)
    max_backtracks_values = list(max_backtracks_values)
    unknown = sorted(set(routines) - set(ROUTINES))
    if unknown:
        raise ValueError(f"Unknown routine(s): {', '.join(unknown)}.")
    if any(not 0.0 < beta < 1.0 for beta in betas):
        raise ValueError("Each beta must be in (0, 1).")
    if any(not 0.0 < sigma < 1.0 for sigma in sigmas):
        raise ValueError("Each sigma must be in (0, 1).")
    if any(initial_step <= 0.0 for initial_step in initial_steps):
        raise ValueError("Each initial step must be positive.")
    if any(max_backtracks < 0 for max_backtracks in max_backtracks_values):
        raise ValueError("Each max_backtracks value must be non-negative.")
    return [
        GridConfig(
            routine=routine,
            beta=beta,
            sigma=sigma,
            initial_step=initial_step,
            max_backtracks=max_backtracks,
        )
        for routine, beta, sigma, initial_step, max_backtracks in product(
            routines,
            betas,
            sigmas,
            initial_steps,
            max_backtracks_values,
        )
    ]


def _as_finite_matrix(name: str, value: object, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a torch.Tensor.")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got ndim={tensor.ndim}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def load_masked_dataset(
    path: str | Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a masked WLRA dataset containing M_full, M_masked, and W tensors."""
    data = torch.load(Path(path), map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("Dataset must be a dictionary.")
    missing = {"M_full", "M_masked", "W"} - set(data)
    if missing:
        raise ValueError(f"Dataset is missing required key(s): {', '.join(sorted(missing))}.")

    a_full = _as_finite_matrix("M_full", data["M_full"], dtype=dtype, device=device)
    a_masked = _as_finite_matrix("M_masked", data["M_masked"], dtype=dtype, device=device)
    w = _as_finite_matrix("W", data["W"], dtype=dtype, device=device)
    if a_full.shape != a_masked.shape or a_full.shape != w.shape:
        raise ValueError(
            "M_full, M_masked, and W must have the same shape. "
            f"Got M_full={tuple(a_full.shape)}, M_masked={tuple(a_masked.shape)}, W={tuple(w.shape)}."
        )
    if not torch.logical_or(w == 0, w == 1).all():
        raise ValueError("W must be binary with entries equal to 0 or 1.")
    return a_full, a_masked, w


def clone_initial_fn(initial_point: Point) -> Callable[[torch.Tensor, int], Point]:
    """Return an initializer that gives every routine/config the same starting point."""
    def _initial_fn(a: torch.Tensor, k: int) -> Point:
        return tuple(block.clone().to(device=a.device, dtype=a.dtype) for block in initial_point)

    return _initial_fn


def _float(value: torch.Tensor) -> float:
    scalar = value.detach().cpu()
    if scalar.numel() != 1:
        raise ValueError("Expected scalar tensor.")
    result = float(scalar.item())
    if not math.isfinite(result):
        raise ValueError("Expected finite scalar value.")
    return result


def _to_float_history(values: Iterable[torch.Tensor]) -> list[float]:
    return [_float(value) for value in values]


def _armijo_evaluation_summary(history: Iterable[int]) -> dict[str, object]:
    """Return JSON-safe summary statistics for per-iteration Armijo checks."""
    values = [int(value) for value in history]
    if any(value < 0 for value in values):
        raise ValueError("Armijo evaluation counts must be non-negative.")
    return {
        "armijo_evaluation_history": values,
        "mean_armijo_evaluations": (sum(values) / len(values)) if values else None,
        "final_armijo_evaluations": values[-1] if values else None,
        "max_armijo_evaluations": max(values) if values else None,
    }


def _rmse_on_missing(point: Point, a_full: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    U, x, V = point
    miss = (1.0 - w).to(dtype=a_full.dtype, device=a_full.device)
    err = miss * (a_full - (U * x.unsqueeze(0)) @ V.transpose(0, 1))
    den = miss.sum().clamp_min(1.0)
    rmse = (err.pow(2).sum() / den).sqrt()
    if not torch.isfinite(rmse):
        raise ValueError("RMSE produced a non-finite value.")
    return rmse


def _auto_radius(a_masked: torch.Tensor, lmbda: float) -> float:
    if lmbda <= 0.0:
        raise ValueError("lmbda must be positive when r='auto'.")
    return _float(torch.linalg.vector_norm(a_masked) / math.sqrt(lmbda))


def resolve_radius(r: float | None, a_masked: torch.Tensor, lmbda: float) -> float:
    """Return the explicit radius or ||a_masked||_F / sqrt(lmbda) when r is None."""
    if r is None:
        return _auto_radius(a_masked, lmbda)
    if not math.isfinite(r):
        raise ValueError("r must be finite.")
    if r < 0.0:
        raise ValueError("r must be non-negative.")
    return float(r)


def _run_regularized_fair_history(
    config: GridConfig,
    *,
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    initial_point: Point,
    k: int,
    iteration_numbers: int,
    lmbda: float,
    armijo_tol: float,
    retraction: str,
    rmse_mode: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], Point, list[int]]:
    """Run regularized Armijo while recording unregularized WLRA loss for comparisons."""
    point = clone_initial_fn(initial_point)(a_masked, k)
    fair_losses: list[torch.Tensor] = []
    regularized_objectives: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []
    armijo_evaluation_history: list[int] = []

    for _ in range(iteration_numbers):
        fair_losses.append(wlra_loss(point, a_masked, w))
        result = armijo_wlra_reg(
            point,
            a_masked,
            w,
            lmbda=lmbda,
            s=config.initial_step,
            beta=config.beta,
            sigma=config.sigma,
            max_backtracks=config.max_backtracks,
            armijo_tol=armijo_tol,
            retraction=retraction,
        )
        armijo_evaluation_history.append(result.armijo_evaluations)
        if not result.accepted:
            error = RuntimeError(f"Armijo step failed: {result.reason}")
            error.armijo_evaluation_history = list(armijo_evaluation_history)
            raise error
        regularized_objectives.append(result.f_current)
        point = result.point_next
        if rmse_mode == "history":
            rmses.append(_rmse_on_missing(point, a_full, w))

    if rmse_mode == "final":
        rmses.append(_rmse_on_missing(point, a_full, w))
    return fair_losses, regularized_objectives, rmses, point, armijo_evaluation_history


def run_config(
    config: GridConfig,
    *,
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    initial_point: Point,
    k: int,
    iteration_numbers: int,
    r: float,
    lmbda: float,
    armijo_tol: float,
    retraction: str,
    rmse_mode: str,
) -> dict[str, object]:
    """Run one grid configuration and return a JSON-serializable record."""
    initial_fn = clone_initial_fn(initial_point)
    regularized_objectives: list[torch.Tensor] | None = None
    armijo_evaluation_history: list[int]
    if config.routine == "armijo_wlra_reg":
        losses, regularized_objectives, rmses, final_point, armijo_evaluation_history = _run_regularized_fair_history(
            config,
            a_full=a_full,
            a_masked=a_masked,
            w=w,
            initial_point=initial_point,
            k=k,
            iteration_numbers=iteration_numbers,
            lmbda=lmbda,
            armijo_tol=armijo_tol,
            retraction=retraction,
            rmse_mode=rmse_mode,
        )
        params = {
            "U": final_point[0],
            "x": final_point[1],
            "V": final_point[2],
        }
    elif config.routine == "armijo_projection_arc":
        diagnostics: dict[str, list[int]] = {}
        try:
            losses, rmses, params, _ = optimize_constraint_armijo(
                a_full,
                a_masked,
                w,
                k=k,
                r=r,
                lr=config.initial_step,
                iteration_numbers=iteration_numbers,
                initial_fn=initial_fn,
                armijo_rule="projection_arc",
                beta=config.beta,
                sigma=config.sigma,
                max_backtracks=config.max_backtracks,
                armijo_tol=armijo_tol,
                retraction=retraction,
                rmse_mode=rmse_mode,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            history = diagnostics.get("armijo_evaluation_history", [])
            if history:
                exc.armijo_evaluation_history = list(history)
            raise
        armijo_evaluation_history = diagnostics.get("armijo_evaluation_history", [])
    elif config.routine == "armijo_feasible_direction":
        diagnostics = {}
        try:
            losses, rmses, params, _ = optimize_constraint_armijo(
                a_full,
                a_masked,
                w,
                k=k,
                r=r,
                lr=config.initial_step,
                iteration_numbers=iteration_numbers,
                initial_fn=initial_fn,
                armijo_rule="feasible_direction",
                beta=config.beta,
                sigma=config.sigma,
                max_backtracks=config.max_backtracks,
                armijo_tol=armijo_tol,
                retraction=retraction,
                rmse_mode=rmse_mode,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            history = diagnostics.get("armijo_evaluation_history", [])
            if history:
                exc.armijo_evaluation_history = list(history)
            raise
        armijo_evaluation_history = diagnostics.get("armijo_evaluation_history", [])
    else:
        raise ValueError(f"Unknown routine: {config.routine}.")

    if len(losses) != iteration_numbers:
        raise ValueError("Optimizer returned an unexpected history length.")
    if len(armijo_evaluation_history) != iteration_numbers:
        raise ValueError("Optimizer returned an unexpected Armijo evaluation history length.")
    if rmse_mode == "history" and len(rmses) != iteration_numbers:
        raise ValueError("Optimizer returned an unexpected RMSE history length.")
    if rmse_mode == "final" and len(rmses) != 1:
        raise ValueError("Optimizer returned an unexpected final RMSE length.")
    final_point = (params["U"], params["x"], params["V"])
    final_wlra_loss = wlra_loss(final_point, a_masked, w)
    record = {
        "config_id": config_id(config, retraction=retraction, rank=k, lmbda=lmbda),
        "status": "success",
        "config": asdict(config),
        "iteration_numbers": iteration_numbers,
        "rank": k,
        "r": r,
        "lmbda": lmbda,
        "armijo_tol": armijo_tol,
        "retraction": retraction,
        "metric_mode": rmse_mode,
        "device": str(a_masked.device),
        "dtype": str(a_masked.dtype).replace("torch.", ""),
        "final_objective": _float(final_wlra_loss),
        "final_wlra_loss": _float(final_wlra_loss),
        "final_data_loss": _float(final_wlra_loss),
        "final_rmse": _float(rmses[-1]),
        "best_objective": min(_float(loss) for loss in losses),
        "best_wlra_loss": min(_float(loss) for loss in losses),
    }
    record.update(_armijo_evaluation_summary(armijo_evaluation_history))
    if regularized_objectives is not None:
        record["final_regularized_objective"] = _float(regularized_objectives[-1])
        record["best_regularized_objective"] = min(_float(loss) for loss in regularized_objectives)
    if rmse_mode == "history":
        record["best_rmse"] = min(_float(rmse) for rmse in rmses)
        record["rmse_history"] = _to_float_history(rmses)
        record["wlra_loss_history"] = _to_float_history(losses)
        if regularized_objectives is not None:
            record["regularized_objective_history"] = _to_float_history(regularized_objectives)
    return record


def write_json_atomic(path: Path, record: dict[str, object]) -> None:
    """Write a checkpoint record atomically so interrupted writes do not corrupt resume state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def load_record(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def record_matches_request(
    record: dict[str, object],
    config: GridConfig,
    *,
    retraction: str,
    rank: int,
    lmbda: float,
    r: float,
    iteration_numbers: int,
    armijo_tol: float,
    rmse_mode: str,
) -> bool:
    """Return whether a checkpoint is for the exact requested experiment config."""
    return (
        record.get("status") == "success"
        and record.get("config") == asdict(config)
        and record.get("retraction") == retraction
        and record.get("rank") == rank
        and math.isclose(float(record.get("lmbda", math.nan)), lmbda, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(float(record.get("r", math.nan)), r, rel_tol=1e-12, abs_tol=1e-12)
        and record.get("iteration_numbers") == iteration_numbers
        and math.isclose(float(record.get("armijo_tol", math.nan)), armijo_tol, rel_tol=1e-12, abs_tol=1e-12)
        and record.get("metric_mode") == rmse_mode
    )


def write_best_summary(raw_dir: Path, summary_path: Path) -> list[dict[str, object]]:
    """Write one best successful configuration per routine/retraction/rank/lambda."""
    records = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name == "grid_config.json":
            continue
        record = load_record(path)
        if record and record.get("status") == "success":
            records.append(record)

    best_by_group: dict[tuple[object, object, object, object], dict[str, object]] = {}
    for record in records:
        routine = record["config"]["routine"]
        group = (routine, record.get("retraction"), record.get("rank"), record.get("lmbda"))
        incumbent = best_by_group.get(group)
        if incumbent is None or (
            record["final_rmse"],
            record["final_wlra_loss"],
        ) < (
            incumbent["final_rmse"],
            incumbent["final_wlra_loss"],
        ):
            best_by_group[group] = record

    rows = []
    for group in sorted(best_by_group):
        routine, retraction, rank, lmbda = group
        record = best_by_group[group]
        config = record["config"]
        rows.append(
            {
                "routine": routine,
                "retraction": retraction,
                "rank": rank,
                "lmbda": lmbda,
                "r": record["r"],
                "config_id": record["config_id"],
                "beta": config["beta"],
                "sigma": config["sigma"],
                "initial_step": config["initial_step"],
                "max_backtracks": config["max_backtracks"],
                "final_rmse": record["final_rmse"],
                "final_wlra_loss": record["final_wlra_loss"],
                "final_data_loss": record["final_data_loss"],
                "final_objective": record["final_objective"],
                "final_regularized_objective": record.get("final_regularized_objective"),
                "best_rmse": record.get("best_rmse"),
                "mean_armijo_evaluations": record.get("mean_armijo_evaluations"),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "routine",
                "retraction",
                "rank",
                "lmbda",
                "r",
                "config_id",
                "beta",
                "sigma",
                "initial_step",
                "max_backtracks",
                "final_rmse",
                "final_wlra_loss",
                "final_data_loss",
                "final_objective",
                "final_regularized_objective",
                "best_rmse",
                "mean_armijo_evaluations",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _load_success_records(raw_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name == "grid_config.json":
            continue
        record = load_record(path)
        if record and record.get("status") == "success":
            records.append(record)
    return records


def write_summary_figures(raw_dir: Path, figures_dir: Path, run_name: str) -> list[Path]:
    """Write summary figures for completed successful configurations."""
    records = _load_success_records(raw_dir)
    if not records:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / f"{run_name}_best_rmse_by_routine.png",
        figures_dir / f"{run_name}_rmse_by_config.png",
    ]

    best_by_routine: dict[str, dict[str, object]] = {}
    for record in records:
        routine = record["config"]["routine"]
        incumbent = best_by_routine.get(routine)
        if incumbent is None or record["final_rmse"] < incumbent["final_rmse"]:
            best_by_routine[routine] = record

    routines = [routine for routine in ROUTINES if routine in best_by_routine]
    best_rmses = [best_by_routine[routine]["final_rmse"] for routine in routines]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(routines, best_rmses)
    ax.set_ylabel("Final RMSE")
    ax.set_title("Best RMSE by Armijo routine")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(paths[0], dpi=200)
    plt.close(fig)

    sorted_records = sorted(records, key=lambda item: (item["config"]["routine"], item["final_rmse"]))
    labels = [f"{item['config']['routine']}:{item['config_id'][-6:]}" for item in sorted_records]
    rmses = [item["final_rmse"] for item in sorted_records]
    fig_width = max(8, min(24, 0.25 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
    ax.plot(range(len(rmses)), rmses, marker="o", linewidth=1.2)
    ax.set_ylabel("Final RMSE")
    ax.set_xlabel("Configuration")
    ax.set_title("Final RMSE for completed grid configurations")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


def _validated_ks(k: int, ks: Iterable[int] | None) -> list[int]:
    values = [k] if ks is None else list(ks)
    if not values:
        raise ValueError("At least one rank k is required.")
    if any(value <= 0 for value in values):
        raise ValueError("Each rank k must be positive.")
    return values


def _validated_lmbdas(lmbda: float, lmbdas: Iterable[float] | None) -> list[float]:
    values = [lmbda] if lmbdas is None else list(lmbdas)
    if not values:
        raise ValueError("At least one lmbda value is required.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Each lmbda value must be finite.")
    return values


def _validated_retractions(retraction: str, retractions: Iterable[str] | None) -> list[str]:
    values = [retraction] if retractions is None else list(retractions)
    if not values:
        raise ValueError("At least one retraction is required.")
    unknown = sorted(set(values) - set(RETRACTIONS))
    if unknown:
        raise ValueError(f"Unknown retraction(s): {', '.join(unknown)}.")
    return values


def write_reproducibility_record(run_processed_dir: Path, metadata: dict[str, object]) -> Path:
    """Write a concise per-run reproducibility record next to processed outputs."""
    path = run_processed_dir / "reproducibility.md"
    lines = [
        f"# Armijo Experiment Reproducibility Record: {metadata['run_id']}",
        "",
        f"- Dataset path: `{metadata['dataset_path']}`",
        f"- Dataset file: `{metadata['dataset_file']}`",
        f"- Routines: {', '.join(str(item) for item in metadata['routines'])}",
        f"- Retractions: {', '.join(str(item) for item in metadata['retractions'])}",
        f"- Ranks: {', '.join(str(item) for item in metadata['ranks'])}",
        f"- Lambdas: {', '.join(str(item) for item in metadata['lmbdas'])}",
        f"- Betas: {', '.join(str(item) for item in metadata['betas'])}",
        f"- Sigmas: {', '.join(str(item) for item in metadata['sigmas'])}",
        f"- Initial steps: {', '.join(str(item) for item in metadata['initial_steps'])}",
        f"- Max backtracks: {', '.join(str(item) for item in metadata['max_backtracks'])}",
        f"- r policy: {metadata['r_policy']}",
        f"- Iterations: {metadata['iteration_numbers']}",
        f"- Armijo tolerance: {metadata['armijo_tol']}",
        f"- Device: {metadata['device']}",
        f"- Dtype: {metadata['dtype']}",
        f"- Metric mode: {metadata['metric_mode']}",
        f"- Raw directory: `{metadata['raw_dir']}`",
        f"- Processed directory: `{metadata['processed_dir']}`",
        f"- Figures directory: `{metadata['figures_dir']}`",
        f"- Created at: {metadata['created_at']}",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def run_grid_search(
    *,
    dataset_path: str | Path,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    run_name: str | None = None,
    routines: Iterable[str] = ROUTINES,
    betas: Iterable[float],
    sigmas: Iterable[float],
    initial_steps: Iterable[float],
    max_backtracks_values: Iterable[int] = (200,),
    k: int = 10,
    ks: Iterable[int] | None = None,
    iteration_numbers: int = 200,
    r: float | None = 100.0,
    lmbda: float = 0.01,
    lmbdas: Iterable[float] | None = None,
    armijo_tol: float = 1e-12,
    retraction: str = "qr",
    retractions: Iterable[str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    force: bool = False,
    rmse_mode: str = "final",
) -> list[dict[str, object]]:
    """Run a checkpointed Armijo grid search and return best summary rows."""
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")
    if rmse_mode not in RMSE_MODES:
        raise ValueError(f"rmse_mode must be one of {', '.join(RMSE_MODES)}.")
    dataset_path = Path(dataset_path)
    raw_root = Path(raw_dir)
    processed_root = Path(processed_dir)
    figures_root = Path(figures_dir)
    run_id = run_name or allocate_run_id(raw_root)
    selected_routines = list(routines)
    selected_betas = list(betas)
    selected_sigmas = list(sigmas)
    selected_initial_steps = list(initial_steps)
    selected_max_backtracks = list(max_backtracks_values)
    selected_ks = _validated_ks(k, ks)
    selected_lmbdas = _validated_lmbdas(lmbda, lmbdas)
    selected_retractions = _validated_retractions(retraction, retractions)
    a_full, a_masked, w = load_masked_dataset(dataset_path, dtype=dtype, device=device)
    configs = build_grid(
        routines=selected_routines,
        betas=selected_betas,
        sigmas=selected_sigmas,
        initial_steps=selected_initial_steps,
        max_backtracks_values=selected_max_backtracks,
    )
    total_configs = len(configs) * len(selected_ks) * len(selected_lmbdas) * len(selected_retractions)
    run_raw_dir = raw_root / run_id
    run_processed_dir = processed_root / run_id
    summary_path = run_processed_dir / "best_configs.csv"
    figures_path = figures_root / run_id
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    run_processed_dir.mkdir(parents=True, exist_ok=True)
    grid_metadata = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_file": dataset_path.name,
        "routines": selected_routines,
        "betas": selected_betas,
        "sigmas": selected_sigmas,
        "initial_steps": selected_initial_steps,
        "max_backtracks": selected_max_backtracks,
        "rank": selected_ks[0] if len(selected_ks) == 1 else None,
        "ranks": selected_ks,
        "iteration_numbers": iteration_numbers,
        "r": r,
        "r_policy": "auto_norm_over_sqrt_lmbda" if r is None else "fixed",
        "lmbda": selected_lmbdas[0] if len(selected_lmbdas) == 1 else None,
        "lmbdas": selected_lmbdas,
        "armijo_tol": armijo_tol,
        "retraction": selected_retractions[0] if len(selected_retractions) == 1 else None,
        "retractions": selected_retractions,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "metric_mode": rmse_mode,
        "total_configs": total_configs,
        "raw_dir": str(run_raw_dir),
        "processed_dir": str(run_processed_dir),
        "figures_dir": str(figures_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_grid_config(grid_config_path(run_raw_dir), grid_metadata)
    write_grid_config(grid_config_path(run_processed_dir), grid_metadata)
    write_reproducibility_record(run_processed_dir, grid_metadata)

    routine_counts: dict[tuple[str, str], int] = {}

    for current_k in selected_ks:
        initial_point = initial_lr(a_masked, current_k)
        for current_retraction in selected_retractions:
            for current_lmbda in selected_lmbdas:
                current_r = resolve_radius(r, a_masked, current_lmbda)
                for config in configs:
                    count_key = (config.routine, current_retraction)
                    routine_counts[count_key] = routine_counts.get(count_key, 0) + 1
                    path = config_path(
                        run_raw_dir,
                        routine=config.routine,
                        retraction=current_retraction,
                        index=routine_counts[count_key],
                    )
                    existing = load_record(path)
                    if (
                        existing
                        and not force
                        and record_matches_request(
                            existing,
                            config,
                            retraction=current_retraction,
                            rank=current_k,
                            lmbda=current_lmbda,
                            r=current_r,
                            iteration_numbers=iteration_numbers,
                            armijo_tol=armijo_tol,
                            rmse_mode=rmse_mode,
                        )
                    ):
                        continue
                    try:
                        record = run_config(
                            config,
                            a_full=a_full,
                            a_masked=a_masked,
                            w=w,
                            initial_point=initial_point,
                            k=current_k,
                            iteration_numbers=iteration_numbers,
                            r=current_r,
                            lmbda=current_lmbda,
                            armijo_tol=armijo_tol,
                            retraction=current_retraction,
                            rmse_mode=rmse_mode,
                        )
                    except Exception as exc:
                        history = getattr(exc, "armijo_evaluation_history", None)
                        failure = {
                            "config_id": config_id(
                                config,
                                retraction=current_retraction,
                                rank=current_k,
                                lmbda=current_lmbda,
                            ),
                            "status": "failed",
                            "config": asdict(config),
                            "run_id": run_id,
                            "config_index": routine_counts[count_key],
                            "total_configs": total_configs,
                            "iteration_numbers": iteration_numbers,
                            "rank": current_k,
                            "r": current_r,
                            "lmbda": current_lmbda,
                            "armijo_tol": armijo_tol,
                            "retraction": current_retraction,
                            "metric_mode": rmse_mode,
                            "device": str(a_masked.device),
                            "dtype": str(a_masked.dtype).replace("torch.", ""),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        if history is not None:
                            failure.update(_armijo_evaluation_summary(history))
                        write_json_atomic(path, failure)
                        write_best_summary(run_raw_dir, summary_path)
                        write_summary_figures(run_raw_dir, figures_path, run_id)
                        continue
                    record["run_id"] = run_id
                    record["config_index"] = routine_counts[count_key]
                    record["total_configs"] = total_configs
                    write_json_atomic(path, record)
                    write_best_summary(run_raw_dir, summary_path)
                    write_summary_figures(run_raw_dir, figures_path, run_id)

    rows = write_best_summary(run_raw_dir, summary_path)
    write_summary_figures(run_raw_dir, figures_path, run_id)
    return rows


def build_arg_parser(
    description: str = "Checkpointed grid search for WLRA Armijo routines.",
    *,
    include_routines: bool = True,
    include_retraction: bool = True,
    include_max_backtracks: bool = True,
    default_run_name: str | None = None,
) -> argparse.ArgumentParser:
    """Build a CLI parser for checkpointed Armijo grid-search runners."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--run-name", default=default_run_name, help="Run id folder name. Defaults to next YYMMDD-NN.")
    if include_routines:
        parser.add_argument("--routines", type=parse_csv_routines, default=list(ROUTINES))
    parser.add_argument("--betas", type=parse_csv_floats, required=True)
    parser.add_argument("--sigmas", type=parse_csv_floats, required=True)
    parser.add_argument("--initial-steps", type=parse_csv_floats, required=True)
    if include_max_backtracks:
        parser.add_argument("--max-backtracks", type=parse_csv_ints, default=[200])
    parser.add_argument("--rank", "-k", type=int, default=10)
    parser.add_argument("--ranks", type=parse_csv_ints, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--r", type=parse_r_value, default=None)
    parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--lmbdas", type=parse_csv_floats, default=None)
    parser.add_argument("--armijo-tol", type=float, default=1e-12)
    if include_retraction:
        parser.add_argument("--retraction", choices=RETRACTIONS, default="qr")
        parser.add_argument("--retractions", type=parse_csv_retractions, default=None)
    parser.add_argument("--rmse-mode", choices=RMSE_MODES, default="final")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def _parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def run_from_args(
    args: argparse.Namespace,
    *,
    routines: Iterable[str] | None = None,
    retraction: str | None = None,
    max_backtracks_values: Iterable[int] | None = None,
) -> None:
    """Run grid search from parsed CLI args, optionally forcing one routine."""
    device = resolve_device(args.device)
    if device.type == "mps" and args.dtype == "float64":
        raise ValueError("MPS does not support float64; use --dtype float32.")
    selected_routines = routines if routines is not None else args.routines
    selected_retraction = retraction if retraction is not None else args.retraction
    selected_retractions = None if retraction is not None else getattr(args, "retractions", None)
    selected_max_backtracks = max_backtracks_values if max_backtracks_values is not None else args.max_backtracks
    run_grid_search(
        dataset_path=args.dataset_path,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        figures_dir=args.figures_dir,
        run_name=args.run_name,
        routines=selected_routines,
        betas=args.betas,
        sigmas=args.sigmas,
        initial_steps=args.initial_steps,
        max_backtracks_values=selected_max_backtracks,
        k=args.rank,
        ks=args.ranks,
        iteration_numbers=args.iterations,
        r=args.r,
        lmbda=args.lmbda,
        lmbdas=args.lmbdas,
        armijo_tol=args.armijo_tol,
        retraction=selected_retraction,
        retractions=selected_retractions,
        dtype=DTYPES[args.dtype],
        device=device,
        force=args.force,
        rmse_mode=args.rmse_mode,
    )


def main_for_routine(routine: str, *, description: str, default_run_name: str) -> None:
    """Run the shared grid-search CLI for a single fixed Armijo routine."""
    parser = build_arg_parser(description, include_routines=False, default_run_name=default_run_name)
    args = parser.parse_args()
    run_from_args(args, routines=[routine])


def main_for_routine_and_retraction(
    routine: str,
    retraction: str,
    *,
    description: str,
    default_run_name: str,
    max_backtracks_values: Iterable[int] | None = None,
) -> None:
    """Run the shared grid-search CLI for one fixed Armijo routine and retraction."""
    parser = build_arg_parser(
        description,
        include_routines=False,
        include_retraction=False,
        include_max_backtracks=max_backtracks_values is None,
        default_run_name=default_run_name,
    )
    args = parser.parse_args()
    run_from_args(args, routines=[routine], retraction=retraction, max_backtracks_values=max_backtracks_values)


def main() -> None:
    run_from_args(_parse_args())


if __name__ == "__main__":
    main()
