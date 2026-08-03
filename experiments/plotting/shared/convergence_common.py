from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from manifold_opt.line_search.armijo import armijo_wlra_reg
from manifold_opt.objectives.losses import wlra_loss
from manifold_opt.optim.gradient_descent import optimize_constraint_armijo
from manifold_opt.optim.initial import initial_lr


DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "mnist0_n600_mask0.70_seed42_train.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "armijo" / "convergence_methods" / "unknown_retraction"
DEFAULT_DTYPE = torch.float32

CurveHistory = dict[str, dict[str, list[float]]]

DISPLAY_NAMES = {
    "projection_arc": "Projection arc",
    "feasible_direction": "Feasible direction",
    "regularized_armijo": "Regularized Armijo",
}
ROUTINE_TO_HISTORY_LABEL = {
    "armijo_feasible_direction": "feasible_direction",
    "armijo_projection_arc": "projection_arc",
    "armijo_wlra_reg": "regularized_armijo",
}
METHOD_ORDER = ("feasible_direction", "projection_arc", "regularized_armijo")
METHOD_COLORS = {
    "feasible_direction": "tab:blue",
    "projection_arc": "tab:orange",
    "regularized_armijo": "tab:green",
}
METHOD_MARKERS = {
    "feasible_direction": "o",
    "projection_arc": "s",
    "regularized_armijo": "^",
}


def _parse_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")


def resolve_experiment_runtime(
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = DEFAULT_DTYPE,
) -> tuple[torch.device, torch.dtype]:
    """Resolve experiment device and dtype, using float32 by default for fair comparisons."""
    if isinstance(device, str) and device == "auto":
        target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    target_dtype = _parse_dtype(dtype)
    if target_device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
        if target_dtype is torch.float64:
            raise ValueError("MPS does not support float64; use dtype='float32'.")
    return target_device, target_dtype


def _as_finite_matrix(name: str, value: object, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a torch.Tensor.")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got ndim={tensor.ndim}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def load_mnist_masked_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
    *,
    dtype: str | torch.dtype = DEFAULT_DTYPE,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the masked MNIST-zero WLRA dataset.

    The file is expected to contain `M_full`, `M_masked`, and binary weights
    `W`, all with the same 2D shape. `W == 0` marks masked entries used for
    RMSE in the optimizer.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    target_device, target_dtype = resolve_experiment_runtime(device=device, dtype=dtype)
    data = torch.load(dataset_path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("Dataset must be a dictionary.")

    missing = {"M_full", "M_masked", "W"} - set(data)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required key(s): {names}.")

    a_full = _as_finite_matrix("M_full", data["M_full"], dtype=target_dtype, device=target_device)
    a_masked = _as_finite_matrix("M_masked", data["M_masked"], dtype=target_dtype, device=target_device)
    w = _as_finite_matrix("W", data["W"], dtype=target_dtype, device=target_device)

    if a_full.shape != a_masked.shape or a_full.shape != w.shape:
        raise ValueError(
            "M_full, M_masked, and W must have the same shape. "
            f"Got M_full={tuple(a_full.shape)}, M_masked={tuple(a_masked.shape)}, W={tuple(w.shape)}."
        )
    if not torch.logical_or(w == 0, w == 1).all():
        raise ValueError("W must be binary with entries equal to 0 or 1.")

    return a_full, a_masked, w


def _clone_initial_fn(point: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Callable:
    def initial_fn(a: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(block.clone().to(device=a.device, dtype=a.dtype) for block in point)

    return initial_fn


def _to_float_list(values: list[torch.Tensor]) -> list[float]:
    result = []
    for value in values:
        scalar = value.detach().cpu()
        if scalar.numel() != 1:
            raise ValueError("Expected scalar history values.")
        result.append(float(scalar.item()))
    return result


def _rmse_on_missing(
    point: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a_full: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    U, x, V = point
    miss = (1.0 - w).to(dtype=a_full.dtype, device=a_full.device)
    err = miss * (a_full - (U * x.unsqueeze(0)) @ V.transpose(0, 1))
    den = miss.sum().clamp_min(1.0)
    rmse = (err.pow(2).sum() / den).sqrt()
    if not torch.isfinite(rmse):
        raise ValueError("RMSE produced a non-finite value.")
    return rmse


def run_armijo_comparison(
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    *,
    retraction: str,
    k: int = 10,
    r: float = 100.0,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
) -> CurveHistory:
    """Run both constrained WLRA Armijo routines from the same initial point."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")

    initial_point = initial_lr(a_masked, k)
    shared_initial_fn = _clone_initial_fn(initial_point)

    histories: CurveHistory = {}
    for label, rule in (
        ("projection_arc", "projection_arc"),
        ("feasible_direction", "feasible_direction"),
    ):
        losses, rmses, _, _ = optimize_constraint_armijo(
            a_full,
            a_masked,
            w,
            k=k,
            r=r,
            lr=lr,
            iteration_numbers=iteration_numbers,
            initial_fn=shared_initial_fn,
            armijo_rule=rule,
            beta=beta,
            sigma=sigma,
            max_backtracks=max_backtracks,
            armijo_tol=armijo_tol,
            retraction=retraction,
        )
        histories[label] = {
            "loss": _to_float_list(losses),
            "rmse": _to_float_list(rmses),
        }

    return histories


def run_regularized_armijo_fair_loss(
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_point: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    k: int = 10,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    retraction: str = "qr",
) -> dict[str, list[float]]:
    """Run regularized Armijo steps while recording unregularized WLRA loss."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")

    point = _clone_initial_fn(initial_point)(a_masked, k)
    losses: list[torch.Tensor] = []
    rmses: list[torch.Tensor] = []

    for _ in range(iteration_numbers):
        losses.append(wlra_loss(point, a_masked, w))
        rmses.append(_rmse_on_missing(point, a_full, w))
        armijo_result = armijo_wlra_reg(
            point,
            a_masked,
            w,
            lmbda=lmbda,
            s=lr,
            beta=beta,
            sigma=sigma,
            max_backtracks=max_backtracks,
            armijo_tol=armijo_tol,
            retraction=retraction,
        )
        if not armijo_result.accepted:
            raise RuntimeError(f"run_regularized_armijo_fair_loss(): Armijo step failed: {armijo_result.reason}")
        point = armijo_result.point_next

    return {
        "loss": _to_float_list(losses),
        "rmse": _to_float_list(rmses),
    }


def run_three_method_comparison(
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    *,
    retraction: str,
    k: int = 10,
    r: float = 100.0,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
) -> CurveHistory:
    """Compare two constrained Armijo rules with regularized Armijo using fair WLRA loss."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if iteration_numbers <= 0:
        raise ValueError("iteration_numbers must be positive.")

    initial_point = initial_lr(a_masked, k)
    shared_initial_fn = _clone_initial_fn(initial_point)
    histories: CurveHistory = {}

    for label, rule in (
        ("feasible_direction", "feasible_direction"),
        ("projection_arc", "projection_arc"),
    ):
        losses, rmses, _, _ = optimize_constraint_armijo(
            a_full,
            a_masked,
            w,
            k=k,
            r=r,
            lr=lr,
            iteration_numbers=iteration_numbers,
            initial_fn=shared_initial_fn,
            armijo_rule=rule,
            beta=beta,
            sigma=sigma,
            max_backtracks=max_backtracks,
            armijo_tol=armijo_tol,
            retraction=retraction,
        )
        histories[label] = {
            "loss": _to_float_list(losses),
            "rmse": _to_float_list(rmses),
        }

    histories["regularized_armijo"] = run_regularized_armijo_fair_loss(
        a_full,
        a_masked,
        w,
        initial_point=initial_point,
        k=k,
        lmbda=lmbda,
        lr=lr,
        iteration_numbers=iteration_numbers,
        beta=beta,
        sigma=sigma,
        max_backtracks=max_backtracks,
        armijo_tol=armijo_tol,
        retraction=retraction,
    )
    return histories


def run_qr_three_method_comparison(
    a_full: torch.Tensor,
    a_masked: torch.Tensor,
    w: torch.Tensor,
    *,
    k: int = 10,
    r: float = 100.0,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
) -> CurveHistory:
    """Compare the two constrained QR Armijo rules with regularized Armijo."""
    return run_three_method_comparison(
        a_full,
        a_masked,
        w,
        retraction="qr",
        k=k,
        r=r,
        lmbda=lmbda,
        lr=lr,
        iteration_numbers=iteration_numbers,
        beta=beta,
        sigma=sigma,
        max_backtracks=max_backtracks,
        armijo_tol=armijo_tol,
    )


def _date_stamp(run_date: date | None = None) -> str:
    """Return a filesystem-safe DD_MM_YYYY date stamp."""
    if run_date is None:
        run_date = date.today()
    return run_date.strftime("%d_%m_%Y")


def _plot_curves(
    history: CurveHistory,
    metric: str,
    ylabel: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, values_by_metric in history.items():
        display = DISPLAY_NAMES.get(label, label.replace("_", " ").title())
        values = values_by_metric[metric]
        iterations = range(1, len(values) + 1)
        ax.plot(iterations, values, label=display, linewidth=1.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_loss_curves(history: CurveHistory, output_path: str | Path, title: str | None = None) -> None:
    """Write the Armijo loss comparison plot."""
    _plot_curves(history, "loss", "Loss", output_path, title=title)


def plot_rmse_curves(history: CurveHistory, output_path: str | Path, title: str | None = None) -> None:
    """Write the Armijo RMSE comparison plot."""
    _plot_curves(history, "rmse", "RMSE on masked entries", output_path, title=title)


def _record_routine(record: dict[str, object]) -> str:
    config = record.get("config")
    if isinstance(config, dict) and config.get("routine") is not None:
        return str(config["routine"])
    return str(record.get("routine", ""))


def _record_history_label(record: dict[str, object]) -> str:
    return ROUTINE_TO_HISTORY_LABEL.get(_record_routine(record), _record_routine(record))


def _record_loss(record: dict[str, object]) -> float:
    for name in ("final_wlra_loss", "final_data_loss", "final_loss", "final_objective"):
        value = record.get(name)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return math.inf


def _as_raw_dirs(raw_dirs: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(raw_dirs, (str, Path)):
        return [Path(raw_dirs)]
    return [Path(raw_dir) for raw_dir in raw_dirs]


def load_curve_records(
    raw_dir: str | Path | Iterable[str | Path],
    *,
    retraction: str | None = None,
) -> list[dict[str, object]]:
    """Load successful history-mode Armijo records for convergence plotting."""
    records = []
    for directory in _as_raw_dirs(raw_dir):
        for path in sorted(directory.rglob("*.json")):
            if path.name == "grid_config.json":
                continue
            record = json.loads(path.read_text())
            if record.get("status") != "success":
                continue
            if record.get("metric_mode") != "history":
                continue
            if retraction is not None and record.get("retraction") != retraction:
                continue
            record["_source_file"] = str(path)
            records.append(record)
    return records


def _history_values(record: dict[str, object], metric: str) -> list[float]:
    if metric == "loss":
        key = "wlra_loss_history"
    elif metric == "rmse":
        key = "rmse_history"
    else:
        raise ValueError("metric must be 'loss' or 'rmse'.")
    values = record.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Record is missing non-empty {key}: {record.get('_source_file', '<memory>')}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{key} contains non-finite values: {record.get('_source_file', '<memory>')}")
    return result


def select_best_curve_records(
    records: Iterable[dict[str, object]],
    *,
    retraction: str,
    ranks: Iterable[int],
    lmbdas: Iterable[float],
) -> dict[tuple[int, float, str], dict[str, object]]:
    """Select the best history record for each rank/lambda/method by final RMSE then fair loss."""
    selected: dict[tuple[int, float, str], dict[str, object]] = {}
    wanted_ranks = set(ranks)
    wanted_lmbdas = list(lmbdas)
    for record in records:
        if record.get("retraction") != retraction:
            continue
        rank = int(record.get("rank", 0))
        if rank not in wanted_ranks:
            continue
        try:
            lmbda = float(record.get("lmbda"))
        except (TypeError, ValueError):
            continue
        if not any(math.isclose(lmbda, value, rel_tol=1e-12, abs_tol=1e-12) for value in wanted_lmbdas):
            continue
        label = _record_history_label(record)
        key = (rank, lmbda, label)
        incumbent = selected.get(key)
        score = (float(record.get("final_rmse", math.inf)), _record_loss(record))
        incumbent_score = (
            float(incumbent.get("final_rmse", math.inf)),
            _record_loss(incumbent),
        ) if incumbent is not None else (math.inf, math.inf)
        if score < incumbent_score:
            selected[key] = record
    return selected


def winning_curve_method(
    selected: dict[tuple[int, float, str], dict[str, object]],
    *,
    rank: int,
    lmbda: float,
) -> str | None:
    """Return the method with lowest final RMSE for one rank/lambda cell."""
    candidates = []
    for method in METHOD_ORDER:
        record = selected.get((rank, lmbda, method))
        if record is None:
            continue
        try:
            rmse = float(record.get("final_rmse", math.inf))
        except (TypeError, ValueError):
            rmse = math.inf
        if not math.isfinite(rmse):
            continue
        candidates.append((rmse, _record_loss(record), method))
    if not candidates:
        return None
    return min(candidates)[2]


def _plot_selected_grid(
    axes: object,
    selected: dict[tuple[int, float, str], dict[str, object]],
    *,
    metric: str,
    rank_values: list[int],
    lmbda_values: list[float],
    font_size: float | None = None,
    publication_layout: bool = False,
    show_legend: bool = True,
) -> None:
    metric_label = "WLRA loss" if metric == "loss" else "RMSE on masked entries"
    for row, rank in enumerate(rank_values):
        for col, lmbda in enumerate(lmbda_values):
            ax = axes[row][col]
            winner = winning_curve_method(selected, rank=rank, lmbda=lmbda)
            for method in METHOD_ORDER:
                record = selected.get((rank, lmbda, method))
                if record is None:
                    continue
                values = _history_values(record, metric)
                iterations = list(range(1, len(values) + 1))
                markevery = max(1, len(values) // 12)
                is_winner = method == winner
                ax.plot(
                    iterations,
                    values,
                    label=DISPLAY_NAMES.get(method, method),
                    color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method),
                    markevery=markevery,
                    linewidth=2.3 if is_winner else 1.2,
                    markersize=3.4 if is_winner else 3.0,
                    zorder=3 if is_winner else 2,
                )
                if is_winner:
                    ax.plot(
                        iterations[-1],
                        values[-1],
                        marker="*",
                        color=METHOD_COLORS.get(method),
                        markeredgecolor="black",
                        markersize=8.0,
                        linestyle="None",
                        zorder=4,
                    )
            ax.set_xscale("log", base=10)
            if font_size is None or not publication_layout:
                ax.set_title(f"p={rank}, lambda={lmbda:g}", fontsize=10)
            else:
                exponent = round(math.log10(lmbda))
                if math.isclose(lmbda, 10.0**exponent):
                    lmbda_label = rf"10^{{{exponent}}}"
                else:
                    lmbda_label = f"{lmbda:g}"
                if row == 0:
                    ax.set_title(
                        rf"$\lambda={lmbda_label}$",
                        fontsize=font_size,
                    )
                ax.tick_params(axis="both", labelsize=font_size)
                ax.xaxis.get_offset_text().set_fontsize(font_size)
                ax.yaxis.get_offset_text().set_fontsize(font_size)
            ax.grid(True, alpha=0.25)
            if row == len(rank_values) - 1:
                ax.set_xlabel("Iteration", fontsize=font_size)
            elif publication_layout:
                ax.tick_params(axis="x", labelbottom=False)
            if publication_layout and col == 0:
                ax.set_ylabel(rf"$p={rank}$", fontsize=font_size)
            elif publication_layout:
                ax.tick_params(axis="y", labelleft=False)
            elif col == 0:
                ax.set_ylabel(metric_label, fontsize=font_size)
            if show_legend and row == 0 and col == 0:
                ax.legend(fontsize=font_size or 8)


def plot_retraction_grid_curves(
    records: Iterable[dict[str, object]],
    *,
    retraction: str,
    metric: str,
    output_path: str | Path,
    ranks: Iterable[int] = (32, 64, 128),
    lmbdas: Iterable[float] = (1e-2, 1e-4, 1e-6),
    title: str | None = None,
) -> Path:
    """Plot a 3x3 convergence grid from history-mode records for one retraction."""
    rank_values = list(ranks)
    lmbda_values = list(lmbdas)
    selected = select_best_curve_records(records, retraction=retraction, ranks=rank_values, lmbdas=lmbda_values)
    if not selected:
        raise ValueError(f"No history records found for retraction={retraction}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(rank_values),
        len(lmbda_values),
        figsize=(4.6 * len(lmbda_values), 3.4 * len(rank_values)),
        sharex=False,
        constrained_layout=True,
        squeeze=False,
    )
    metric_label = "WLRA loss" if metric == "loss" else "RMSE on masked entries"
    fig.suptitle(title or f"{retraction.upper()} Armijo {metric_label} convergence", fontsize=14)
    _plot_selected_grid(
        axes,
        selected,
        metric=metric,
        rank_values=rank_values,
        lmbda_values=lmbda_values,
    )

    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_merged_grid_convergence_curves(
    datasets: Iterable[tuple[str, Iterable[dict[str, object]]]],
    *,
    retractions: Iterable[str],
    metric: str,
    output_path: str | Path,
    ranks: Iterable[int] = (32, 64, 128),
    lmbdas: Iterable[float] = (1e-2, 1e-4, 1e-6),
    font_size: float = 34.0,
) -> Path:
    """Plot dataset rows and retraction columns as one convergence figure."""
    dataset_values = [(label, list(records)) for label, records in datasets]
    if not dataset_values:
        raise ValueError("At least one labeled dataset is required.")
    retraction_values = list(retractions)
    if not retraction_values:
        raise ValueError("At least one retraction is required.")
    if metric not in {"loss", "rmse"}:
        raise ValueError("metric must be 'loss' or 'rmse'.")
    if not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be a positive finite value.")

    rank_values = list(ranks)
    lmbda_values = list(lmbdas)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(
        figsize=(
            4.4 * len(lmbda_values) * len(retraction_values),
            3.5 * len(rank_values) * len(dataset_values),
        ),
        constrained_layout=True,
    )
    dataset_subfigures = fig.subfigures(
        nrows=len(dataset_values), ncols=1, squeeze=False
    )
    for dataset_row, (dataset_label, records) in enumerate(dataset_values):
        dataset_subfigure = dataset_subfigures[dataset_row][0]
        dataset_subfigure.suptitle(
            dataset_label,
            fontsize=1.15 * font_size,
            fontweight="bold",
        )
        retraction_subfigures = dataset_subfigure.subfigures(
            nrows=1,
            ncols=len(retraction_values),
            squeeze=False,
        )
        for retraction_col, retraction in enumerate(retraction_values):
            selected = select_best_curve_records(
                records,
                retraction=retraction,
                ranks=rank_values,
                lmbdas=lmbda_values,
            )
            if not selected:
                raise ValueError(
                    f"No history records found for dataset={dataset_label}, retraction={retraction}."
                )
            retraction_subfigure = retraction_subfigures[0][retraction_col]
            retraction_subfigure.suptitle(
                f"{retraction.upper()} retraction",
                fontsize=1.05 * font_size,
            )
            axes = retraction_subfigure.subplots(
                len(rank_values),
                len(lmbda_values),
                sharex=False,
                squeeze=False,
            )
            _plot_selected_grid(
                axes,
                selected,
                metric=metric,
                rank_values=rank_values,
                lmbda_values=lmbda_values,
                font_size=font_size,
                publication_layout=True,
                show_legend=False,
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncols=len(METHOD_ORDER),
        fontsize=font_size,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def create_grid_convergence_figures(
    *,
    raw_dir: str | Path | None = None,
    raw_dirs: Iterable[str | Path] | None = None,
    output_dir: str | Path,
    retractions: Iterable[str] = ("qr", "polar"),
    ranks: Iterable[int] = (32, 64, 128),
    lmbdas: Iterable[float] = (1e-2, 1e-4, 1e-6),
    metrics: Iterable[str] = ("loss", "rmse"),
    run_name: str | None = None,
) -> list[Path]:
    """Create loss and RMSE 3x3 figures from completed history-mode grid records."""
    output = Path(output_dir)
    selected_raw_dirs: list[str | Path] = []
    if raw_dir is not None:
        selected_raw_dirs.append(raw_dir)
    if raw_dirs is not None:
        selected_raw_dirs.extend(raw_dirs)
    if not selected_raw_dirs:
        raise ValueError("At least one raw directory is required.")
    metric_values = list(metrics)
    unknown_metrics = sorted(set(metric_values) - {"loss", "rmse"})
    if unknown_metrics:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown_metrics)}.")
    records = load_curve_records(selected_raw_dirs)
    suffix = run_name or _date_stamp()
    paths: list[Path] = []
    for retraction in retractions:
        for metric in metric_values:
            paths.append(
                plot_retraction_grid_curves(
                    records,
                    retraction=retraction,
                    metric=metric,
                    output_path=output / f"armijo_{retraction}_{metric}_grid_convergence_{suffix}.png",
                    ranks=ranks,
                    lmbdas=lmbdas,
                )
            )
    return paths


def convergence_title(metric: str, *, retraction: str, lmbda: float, r: float, beta: float, sigma: float) -> str:
    """Return a plot title containing the Armijo experiment parameters."""
    return (
        f"{metric} convergence "
        f"(retraction={retraction}, lambda={lmbda:g}, r={r:g}, beta={beta:g}, sigma={sigma:g})"
    )


def create_retraction_convergence_plots(
    retraction: str,
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k: int = 10,
    r: float = 100.0,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    dtype: str | torch.dtype = DEFAULT_DTYPE,
    device: str | torch.device = "cpu",
    run_date: date | None = None,
) -> CurveHistory:
    """Load the dataset, run one retraction comparison, and save timestamped plots."""
    a_full, a_masked, w = load_mnist_masked_dataset(dataset_path, dtype=dtype, device=device)
    history = run_armijo_comparison(
        a_full,
        a_masked,
        w,
        retraction=retraction,
        k=k,
        r=r,
        lr=lr,
        iteration_numbers=iteration_numbers,
        beta=beta,
        sigma=sigma,
        max_backtracks=max_backtracks,
        armijo_tol=armijo_tol,
    )

    output = Path(output_dir)
    stamp = _date_stamp(run_date)
    plot_loss_curves(history, output / f"armijo_{retraction}_loss_convergence_{stamp}.png")
    plot_rmse_curves(history, output / f"armijo_{retraction}_rmse_convergence_{stamp}.png")
    return history


def create_qr_three_method_convergence_plots(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k: int = 10,
    r: float = 100.0,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    dtype: str | torch.dtype = DEFAULT_DTYPE,
    device: str | torch.device = "cpu",
    run_date: date | None = None,
) -> CurveHistory:
    """Load the dataset, compare three QR methods, and save timestamped plots."""
    a_full, a_masked, w = load_mnist_masked_dataset(dataset_path, dtype=dtype, device=device)
    history = run_qr_three_method_comparison(
        a_full,
        a_masked,
        w,
        k=k,
        r=r,
        lmbda=lmbda,
        lr=lr,
        iteration_numbers=iteration_numbers,
        beta=beta,
        sigma=sigma,
        max_backtracks=max_backtracks,
        armijo_tol=armijo_tol,
    )

    output = Path(output_dir)
    stamp = _date_stamp(run_date)
    plot_loss_curves(
        history,
        output / f"armijo_qr_loss_convergence_{stamp}.png",
        title=convergence_title("Loss", retraction="qr", lmbda=lmbda, r=r, beta=beta, sigma=sigma),
    )
    plot_rmse_curves(
        history,
        output / f"armijo_qr_rmse_convergence_{stamp}.png",
        title=convergence_title("RMSE", retraction="qr", lmbda=lmbda, r=r, beta=beta, sigma=sigma),
    )
    return history


def create_polar_three_method_convergence_plots(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k: int = 10,
    r: float = 100.0,
    lmbda: float = 0.01,
    lr: float = 0.01,
    iteration_numbers: int = 100,
    beta: float = 0.5,
    sigma: float = 0.25,
    max_backtracks: int = 50,
    armijo_tol: float = 1e-12,
    dtype: str | torch.dtype = DEFAULT_DTYPE,
    device: str | torch.device = "cpu",
    run_date: date | None = None,
) -> CurveHistory:
    """Load the dataset, compare three polar methods, and save timestamped plots."""
    a_full, a_masked, w = load_mnist_masked_dataset(dataset_path, dtype=dtype, device=device)
    history = run_three_method_comparison(
        a_full,
        a_masked,
        w,
        retraction="polar",
        k=k,
        r=r,
        lmbda=lmbda,
        lr=lr,
        iteration_numbers=iteration_numbers,
        beta=beta,
        sigma=sigma,
        max_backtracks=max_backtracks,
        armijo_tol=armijo_tol,
    )

    output = Path(output_dir)
    stamp = _date_stamp(run_date)
    plot_loss_curves(
        history,
        output / f"armijo_polar_loss_convergence_{stamp}.png",
        title=convergence_title("Loss", retraction="polar", lmbda=lmbda, r=r, beta=beta, sigma=sigma),
    )
    plot_rmse_curves(
        history,
        output / f"armijo_polar_rmse_convergence_{stamp}.png",
        title=convergence_title("RMSE", retraction="polar", lmbda=lmbda, r=r, beta=beta, sigma=sigma),
    )
    return history


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", "-k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--r", type=float, default=100.0)
    parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--max-backtracks", type=int, default=50)
    parser.add_argument("--armijo-tol", type=float, default=1e-12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def create_from_args(retraction: str, args: argparse.Namespace) -> CurveHistory:
    return create_retraction_convergence_plots(
        retraction,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        k=args.rank,
        r=args.r,
        lr=args.lr,
        iteration_numbers=args.iterations,
        beta=args.beta,
        sigma=args.sigma,
        max_backtracks=args.max_backtracks,
        armijo_tol=args.armijo_tol,
        device=args.device,
        dtype=args.dtype,
    )
