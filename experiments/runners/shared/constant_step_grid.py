from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.projection import feasible_set_projection
from manifold_opt.geometry.riemannian_gradient import grad, grad_reg
from manifold_opt.objectives.losses import wlra_loss, wlra_loss_reg
from manifold_opt.optim.gradient_descent import _resolve_constant_retraction
from manifold_opt.optim.initial import initial_lr, initialization_method_for_input
from experiments.runners.shared.armijo_grid import DTYPES, DEFAULT_DATASET_PATH, _float
from experiments.runners.shared.armijo_grid import load_masked_dataset, resolve_device, write_json_atomic


DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "constant_step"
RETRACTIONS = ("qr", "polar")
Routine = Literal["optimize_reg", "optimize_constraint_constant"]
Point = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ConstantStepConfig:
    learning_rate: float


def parse_csv_learning_rates(value: str) -> list[float]:
    """Parse positive finite learning rates from a comma-separated string."""
    learning_rates = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not learning_rates:
        raise ValueError("Expected at least one learning rate.")
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in learning_rates):
        raise ValueError("Each learning rate must be positive and finite.")
    return learning_rates


def build_grid(*, learning_rates: list[float]) -> list[ConstantStepConfig]:
    """Build the constant-stepsize grid over learning rates."""
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in learning_rates):
        raise ValueError("Each learning rate must be positive and finite.")
    return [ConstantStepConfig(learning_rate=rate) for rate in learning_rates]


def clone_initial_point(initial_point: Point, *, a: torch.Tensor) -> Point:
    """Clone a shared initial point onto the target tensor dtype and device."""
    return tuple(block.clone().to(device=a.device, dtype=a.dtype) for block in initial_point)


def rmse_on_missing(point: Point, a_full: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute RMSE only on missing entries marked by w == 0."""
    U, x, V = point
    miss = (1.0 - w).to(dtype=a_full.dtype, device=a_full.device)
    err = miss * (a_full - (U * x.unsqueeze(0)) @ V.transpose(0, 1))
    den = miss.sum().clamp_min(1.0)
    rmse = (err.pow(2).sum() / den).sqrt()
    if not torch.isfinite(rmse):
        raise ValueError("RMSE produced a non-finite value.")
    return rmse


def tangent_norm(point: Point, tangent: Point) -> torch.Tensor:
    """Compute the Riemannian norm of a tangent vector."""
    norm = torch.sqrt(inner_product(point, tangent, tangent).clamp_min(0.0))
    if not torch.isfinite(norm):
        raise ValueError("Gradient norm produced a non-finite value.")
    return norm


def _to_float_history(values: list[torch.Tensor]) -> list[float]:
    return [_float(value) for value in values]


def _write_curve_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _run_optimize_reg(
    config: ConstantStepConfig,
    *,
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    initial_point: Point,
    k: int,
    iteration_numbers: int,
    lmbda: float,
    retraction: str,
) -> tuple[dict[str, object], dict[str, object]]:
    point = clone_initial_point(initial_point, a=a_masked)
    retract = _resolve_constant_retraction(retraction)
    gradient = tuple(torch.zeros_like(block) for block in point)
    data_losses: list[torch.Tensor] = []
    objectives: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []

    for _ in range(iteration_numbers):
        gradient = grad_reg(point, a_masked, w, lmbda)
        data_losses.append(wlra_loss(point, a_masked, w))
        objectives.append(wlra_loss_reg(point, a_masked, w, lmbda))
        step = tuple(-config.learning_rate * block for block in gradient)
        point = retract(point, step)
        rmses.append(rmse_on_missing(point, a_full, w))

    final_gradient = grad_reg(point, a_masked, w, lmbda)
    final_loss = wlra_loss(point, a_masked, w)
    final_objective = wlra_loss_reg(point, a_masked, w, lmbda)
    summary = {
        "final_rmse": _float(rmse_on_missing(point, a_full, w)),
        "final_loss": _float(final_loss),
        "best_rmse": min(_to_float_history(rmses)),
        "best_loss": min(_to_float_history(data_losses)),
        "final_objective": _float(final_objective),
        "final_gradient_norm": _float(tangent_norm(point, final_gradient)),
        "final_x_norm": _float(torch.linalg.vector_norm(point[1])),
    }
    curve = {
        "loss_history": _to_float_history(data_losses),
        "rmse_history": _to_float_history(rmses),
        "objective_history": _to_float_history(objectives),
    }
    return summary, curve


def _run_optimize_constraint_constant(
    config: ConstantStepConfig,
    *,
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    initial_point: Point,
    k: int,
    iteration_numbers: int,
    r: float,
    retraction: str,
) -> tuple[dict[str, object], dict[str, object]]:
    point = clone_initial_point(initial_point, a=a_masked)
    retract = _resolve_constant_retraction(retraction)
    gradient = tuple(torch.zeros_like(block) for block in point)
    losses: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []

    for _ in range(iteration_numbers):
        gradient = grad(point, a_masked, w)
        losses.append(wlra_loss(point, a_masked, w))
        descent = tuple(-config.learning_rate * block for block in gradient)
        step = feasible_set_projection(point, descent, r)
        point = retract(point, step)
        rmses.append(rmse_on_missing(point, a_full, w))

    final_gradient = grad(point, a_masked, w)
    final_loss = wlra_loss(point, a_masked, w)
    summary = {
        "final_rmse": _float(rmse_on_missing(point, a_full, w)),
        "final_loss": _float(final_loss),
        "best_rmse": min(_to_float_history(rmses)),
        "best_loss": min(_to_float_history(losses)),
        "final_gradient_norm": _float(tangent_norm(point, final_gradient)),
        "final_x_norm": _float(torch.linalg.vector_norm(point[1])),
    }
    curve = {
        "loss_history": _to_float_history(losses),
        "rmse_history": _to_float_history(rmses),
    }
    return summary, curve


def run_config(
    routine: Routine,
    config: ConstantStepConfig,
    *,
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    initial_point: Point,
    k: int,
    iteration_numbers: int,
    r: float,
    lmbda: float,
    retraction: str,
    initialization_method: str,
    curve_path: Path,
) -> dict[str, object]:
    """Run one constant-stepsize config and write its curve file."""
    if routine == "optimize_reg":
        summary, curve = _run_optimize_reg(
            config,
            a_full=a_full,
            a_masked=a_masked,
            w=w,
            initial_point=initial_point,
            k=k,
            iteration_numbers=iteration_numbers,
            lmbda=lmbda,
            retraction=retraction,
        )
    elif routine == "optimize_constraint_constant":
        summary, curve = _run_optimize_constraint_constant(
            config,
            a_full=a_full,
            a_masked=a_masked,
            w=w,
            initial_point=initial_point,
            k=k,
            iteration_numbers=iteration_numbers,
            r=r,
            retraction=retraction,
        )
    else:
        raise ValueError(f"Unknown routine: {routine}.")

    curve_payload = {
        "routine": routine,
        "retraction": retraction,
        "config": asdict(config),
        "iteration_numbers": iteration_numbers,
        **curve,
    }
    _write_curve_file(curve_path, curve_payload)
    return {
        "status": "success",
        "routine": routine,
        "retraction": retraction,
        "config": asdict(config),
        "iteration_numbers": iteration_numbers,
        "rank": k,
        "r": r,
        "lmbda": lmbda,
        "device": str(a_masked.device),
        "dtype": str(a_masked.dtype).replace("torch.", ""),
        "initialization": initialization_method,
        "curve_file": str(curve_path),
        **summary,
    }


def run_grid_search(
    *,
    routine: Routine,
    dataset_path: Path,
    raw_dir: Path = DEFAULT_RAW_DIR,
    learning_rates: list[float],
    k: int = 10,
    iteration_numbers: int = 100,
    r: float = 100.0,
    lmbda: float = 0.01,
    retraction: str = "qr",
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    run_timestamp: str | None = None,
) -> list[Path]:
    """Run a constant-stepsize grid and write per-config JSON files."""
    if routine not in {"optimize_reg", "optimize_constraint_constant"}:
        raise ValueError(f"Unknown routine: {routine}.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")
    if r < 0:
        raise ValueError("r must be non-negative.")
    if not math.isfinite(lmbda):
        raise ValueError("lmbda must be finite.")
    _resolve_constant_retraction(retraction)

    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    a_full, a_masked, w = load_masked_dataset(dataset_path, dtype=dtype, device=device)
    initialization_method = initialization_method_for_input(a_masked)
    initial_point = initial_lr(a_masked, k)
    configs = build_grid(learning_rates=learning_rates)

    raw_dir.mkdir(parents=True, exist_ok=True)
    curve_dir = raw_dir / "curves"
    paths = []
    for index, config in enumerate(configs, start=1):
        stem = f"{routine}_{retraction}_{timestamp}_lr{config.learning_rate:g}"
        path = raw_dir / f"{stem}.json"
        curve_path = curve_dir / f"{stem}_curve.json"
        try:
            record = run_config(
                routine,
                config,
                a_full=a_full,
                a_masked=a_masked,
                w=w,
                initial_point=initial_point,
                k=k,
                iteration_numbers=iteration_numbers,
                r=r,
                lmbda=lmbda,
                retraction=retraction,
                initialization_method=initialization_method,
                curve_path=curve_path,
            )
        except Exception as exc:
            record = {
                "status": "failed",
                "routine": routine,
                "retraction": retraction,
                "config": asdict(config),
                "iteration_numbers": iteration_numbers,
                "rank": k,
                "r": r,
                "lmbda": lmbda,
                "device": str(a_masked.device),
                "dtype": str(a_masked.dtype).replace("torch.", ""),
                "initialization": initialization_method,
                "curve_file": str(curve_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        record["run_timestamp"] = timestamp
        record["config_index"] = index
        record["total_configs"] = len(configs)
        write_json_atomic(path, record)
        paths.append(path)
    return paths


def build_arg_parser(routine: Routine, description: str) -> argparse.ArgumentParser:
    """Build a CLI parser for a constant-stepsize runner."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--learning-rates", type=parse_csv_learning_rates, required=True)
    parser.add_argument("--rank", "-k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    if routine == "optimize_reg":
        parser.add_argument("--lmbda", type=float, default=0.01)
        parser.set_defaults(r=100.0)
    else:
        parser.add_argument("--r", type=float, default=100.0)
        parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--retraction", choices=RETRACTIONS, default="qr")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    return parser


def main_for_routine(routine: Routine, *, description: str) -> None:
    """Run a fixed constant-stepsize routine from CLI args."""
    args = build_arg_parser(routine, description).parse_args()
    device = resolve_device(args.device)
    if device.type == "mps" and args.dtype == "float64":
        raise ValueError("MPS does not support float64; use --dtype float32.")
    run_grid_search(
        routine=routine,
        dataset_path=args.dataset_path,
        raw_dir=args.raw_dir,
        learning_rates=args.learning_rates,
        k=args.rank,
        iteration_numbers=args.iterations,
        r=args.r,
        lmbda=args.lmbda,
        retraction=args.retraction,
        dtype=DTYPES[args.dtype],
        device=device,
    )
