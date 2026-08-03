from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manifold_opt.geometry.inner_product import inner_product
from manifold_opt.geometry.riemannian_gradient import grad
from manifold_opt.line_search.armijo import armijo_feasible_direction
from manifold_opt.objectives.losses import wlra_loss
from manifold_opt.optim.initial import initial_lr, initialization_method_for_input
from experiments.runners.shared.armijo_grid import DTYPES, DEFAULT_DATASET_PATH, RMSE_MODES, _float, load_masked_dataset
from experiments.runners.shared.armijo_grid import parse_csv_floats
from experiments.runners.shared.armijo_grid import resolve_device, write_json_atomic


ROUTINE = "armijo_feasible_direction"
RETRACTION = "qr"
INITIAL_STEPS = (0.1, 0.3, 1.0)
MAX_BACKTRACKS = 200
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "armijo" / "feasible_direction" / "qr"

Point = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FeasibleDirectionQrConfig:
    beta: float
    sigma: float
    initial_step: float


def build_grid(
    *,
    betas: list[float],
    sigmas: list[float],
    initial_steps: list[float] = list(INITIAL_STEPS),
) -> list[FeasibleDirectionQrConfig]:
    """Build the feasible-direction QR grid over beta, sigma, and s_bar."""
    if any(not 0.0 < beta < 1.0 for beta in betas):
        raise ValueError("Each beta must be in (0, 1).")
    if any(not 0.0 < sigma < 1.0 for sigma in sigmas):
        raise ValueError("Each sigma must be in (0, 1).")
    if any(initial_step <= 0.0 for initial_step in initial_steps):
        raise ValueError("Each initial step must be positive.")
    return [
        FeasibleDirectionQrConfig(beta=beta, sigma=sigma, initial_step=initial_step)
        for beta, sigma, initial_step in product(betas, sigmas, initial_steps)
    ]


def clone_initial_fn(initial_point: Point):
    """Return an initializer that gives every config the same starting point."""
    def _initial_fn(a: torch.Tensor, k: int) -> Point:
        return tuple(block.clone().to(device=a.device, dtype=a.dtype) for block in initial_point)

    return _initial_fn


def rmse_on_missing(point: Point, a_full: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    U, x, V = point
    miss = (1.0 - w).to(dtype=a_full.dtype, device=a_full.device)
    err = miss * (a_full - (U * x.unsqueeze(0)) @ V.transpose(0, 1))
    den = miss.sum().clamp_min(1.0)
    rmse = (err.pow(2).sum() / den).sqrt()
    if not torch.isfinite(rmse):
        raise ValueError("RMSE produced a non-finite value.")
    return rmse


def tangent_norm(point: Point, tangent: Point) -> torch.Tensor:
    norm = torch.sqrt(inner_product(point, tangent, tangent).clamp_min(0.0))
    if not torch.isfinite(norm):
        raise ValueError("Gradient norm produced a non-finite value.")
    return norm


def run_config(
    config: FeasibleDirectionQrConfig,
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
    initialization_method: str,
    rmse_mode: str = "final",
) -> dict[str, object]:
    """Run one feasible-direction QR config and return a JSON-serializable record."""
    if rmse_mode not in RMSE_MODES:
        raise ValueError(f"rmse_mode must be one of {', '.join(RMSE_MODES)}.")
    point = clone_initial_fn(initial_point)(a_masked, k)
    max_backtracks_hit = False
    last_m = 0
    rmse_history: list[torch.Tensor] = []

    for _ in range(iteration_numbers):
        result = armijo_feasible_direction(
            point,
            a_masked,
            w,
            r=r,
            s=config.initial_step,
            beta=config.beta,
            sigma=config.sigma,
            max_backtracks=MAX_BACKTRACKS,
            armijo_tol=armijo_tol,
            retraction=RETRACTION,
        )
        if not result.accepted:
            raise RuntimeError(f"Armijo step failed: {result.reason}")
        max_backtracks_hit = max_backtracks_hit or result.m >= MAX_BACKTRACKS
        last_m = result.m
        point = result.point_next
        if rmse_mode == "history":
            rmse_history.append(rmse_on_missing(point, a_full, w))

    final_gradient = grad(point, a_masked, w)
    final_loss = wlra_loss(point, a_masked, w)
    final_rmse = rmse_history[-1] if rmse_history else rmse_on_missing(point, a_full, w)
    record = {
        "status": "success",
        "routine": ROUTINE,
        "retraction": RETRACTION,
        "config": asdict(config) | {"max_backtracks": MAX_BACKTRACKS},
        "iteration_numbers": iteration_numbers,
        "rank": k,
        "r": r,
        "lmbda": lmbda,
        "armijo_tol": armijo_tol,
        "device": str(a_masked.device),
        "dtype": str(a_masked.dtype).replace("torch.", ""),
        "initialization": initialization_method,
        "metric_mode": rmse_mode,
        "final_rmse": _float(final_rmse),
        "final_loss": _float(final_loss),
        "final_gradient_norm": _float(tangent_norm(point, final_gradient)),
        "final_x_norm": _float(torch.linalg.vector_norm(point[1])),
        "max_backtracks_hit": max_backtracks_hit,
        "final_backtracks": last_m,
    }
    if rmse_mode == "history":
        record["best_rmse"] = min(_float(rmse) for rmse in rmse_history)
        record["rmse_history"] = [_float(rmse) for rmse in rmse_history]
    return record


def run_grid_search(
    *,
    dataset_path: Path,
    raw_dir: Path = DEFAULT_RAW_DIR,
    betas: list[float],
    sigmas: list[float],
    initial_steps: list[float] = list(INITIAL_STEPS),
    k: int = 10,
    iteration_numbers: int = 100,
    r: float = 100.0,
    lmbda: float = 0.01,
    armijo_tol: float = 1e-12,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    run_timestamp: str | None = None,
    rmse_mode: str = "final",
) -> list[Path]:
    """Run the feasible-direction QR grid and write timestamped per-config JSON files."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")
    if r < 0:
        raise ValueError("r must be non-negative.")
    if not math.isfinite(lmbda):
        raise ValueError("lmbda must be finite.")
    if rmse_mode not in RMSE_MODES:
        raise ValueError(f"rmse_mode must be one of {', '.join(RMSE_MODES)}.")

    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    a_full, a_masked, w = load_masked_dataset(dataset_path, dtype=dtype, device=device)
    initialization_method = initialization_method_for_input(a_masked)
    initial_point = initial_lr(a_masked, k)
    configs = build_grid(betas=betas, sigmas=sigmas, initial_steps=initial_steps)

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, config in enumerate(configs, start=1):
        stem = (
            f"{ROUTINE}_{RETRACTION}_{timestamp}_"
            f"beta{config.beta:g}_sigma{config.sigma:g}_sbar{config.initial_step:g}"
        )
        path = raw_dir / f"{stem}.json"
        try:
            record = run_config(
                config,
                a_full=a_full,
                a_masked=a_masked,
                w=w,
                initial_point=initial_point,
                k=k,
                iteration_numbers=iteration_numbers,
                r=r,
                lmbda=lmbda,
                armijo_tol=armijo_tol,
                initialization_method=initialization_method,
                rmse_mode=rmse_mode,
            )
        except Exception as exc:
            record = {
                "status": "failed",
                "routine": ROUTINE,
                "retraction": RETRACTION,
                "config": asdict(config) | {"max_backtracks": MAX_BACKTRACKS},
                "iteration_numbers": iteration_numbers,
                "rank": k,
                "r": r,
                "lmbda": lmbda,
                "armijo_tol": armijo_tol,
                "device": str(a_masked.device),
                "dtype": str(a_masked.dtype).replace("torch.", ""),
                "initialization": initialization_method,
                "metric_mode": rmse_mode,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        record["run_timestamp"] = timestamp
        record["config_index"] = index
        record["total_configs"] = len(configs)
        write_json_atomic(path, record)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timestamped raw JSON grid search for feasible-direction Armijo with QR retraction."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--betas", type=parse_csv_floats, required=True)
    parser.add_argument("--sigmas", type=parse_csv_floats, required=True)
    parser.add_argument("--initial-steps", type=parse_csv_floats, default=list(INITIAL_STEPS))
    parser.add_argument("--rank", "-k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--r", type=float, default=100.0)
    parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--armijo-tol", type=float, default=1e-12)
    parser.add_argument("--rmse-mode", choices=RMSE_MODES, default="final")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "mps" and args.dtype == "float64":
        raise ValueError("MPS does not support float64; use --dtype float32.")
    run_grid_search(
        dataset_path=args.dataset_path,
        raw_dir=args.raw_dir,
        betas=args.betas,
        sigmas=args.sigmas,
        initial_steps=args.initial_steps,
        k=args.rank,
        iteration_numbers=args.iterations,
        r=args.r,
        lmbda=args.lmbda,
        armijo_tol=args.armijo_tol,
        dtype=DTYPES[args.dtype],
        device=device,
        rmse_mode=args.rmse_mode,
    )


if __name__ == "__main__":
    main()
