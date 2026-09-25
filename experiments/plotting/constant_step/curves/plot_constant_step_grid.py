from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "constant_step"
DEFAULT_PROCESSED_DIR = (
    REPO_ROOT / "results" / "figures" / "constant_step" / "optimize_reg_vs_optimize_constraint_constant" / "qr"
)
ROUTINES = ("optimize_reg", "optimize_constraint_constant")


def _format_float(value: float) -> str:
    return f"{value:.10g}"


def _filename_float(value: float) -> str:
    text = f"{value:.0e}" if abs(value) < 1e-3 or abs(value) >= 1e3 else f"{value:g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    record["_source_file"] = str(path)
    return record


def load_records(raw_dirs: list[Path], pattern: str) -> list[dict[str, Any]]:
    """Load constant-step JSON records from raw_dirs, excluding curve files."""
    records = []
    seen_paths = set()
    for raw_dir in raw_dirs:
        for path in sorted(raw_dir.rglob(pattern)):
            if "curves" in path.parts or path in seen_paths:
                continue
            seen_paths.add(path)
            records.append(_load_json(path))
    return records


def _value(record: dict[str, Any], name: str) -> Any:
    config = record.get("config", {})
    if name in config:
        return config[name]
    return record.get(name)


def _as_float(record: dict[str, Any], name: str) -> float:
    value = _value(record, name)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite in {record.get('_source_file')}.")
    return number


def _record_key(record: dict[str, Any]) -> tuple[str, float, float]:
    return (
        str(record.get("routine")),
        _as_float(record, "learning_rate"),
        _as_float(record, "lmbda"),
    )


def successful_records(records: list[dict[str, Any]], retraction: str) -> list[dict[str, Any]]:
    """Return successful records for the two constant-step routines and one retraction."""
    return [
        record
        for record in records
        if record.get("status") == "success"
        and record.get("routine") in ROUTINES
        and record.get("retraction") == retraction
    ]


def _index_records(records: list[dict[str, Any]]) -> dict[tuple[str, float, float], dict[str, Any]]:
    indexed = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise ValueError(
                "Duplicate constant-step record for "
                f"routine={key[0]}, learning_rate={key[1]}, lmbda={key[2]}: "
                f"{indexed[key].get('_source_file')} and {record.get('_source_file')}"
            )
        indexed[key] = record
    return indexed


def _curve_path(record: dict[str, Any]) -> Path:
    curve_file = record.get("curve_file")
    if not curve_file:
        raise ValueError(f"Missing curve_file in {record.get('_source_file')}.")
    path = Path(str(curve_file))
    if path.exists():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    source_relative = Path(str(record["_source_file"])).parent / path
    if source_relative.exists():
        return source_relative
    raise FileNotFoundError(f"Curve file not found for {record.get('_source_file')}: {curve_file}")


def _load_curve(record: dict[str, Any]) -> dict[str, list[float]]:
    curve = json.loads(_curve_path(record).read_text())
    for name in ("loss_history", "rmse_history"):
        values = curve.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list in {record.get('curve_file')}.")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} contains non-finite values in {record.get('curve_file')}.")
    return curve


def _require_pair(
    indexed: dict[tuple[str, float, float], dict[str, Any]],
    *,
    learning_rate: float,
    lmbda: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = []
    for routine in ROUTINES:
        key = (routine, learning_rate, lmbda)
        record = indexed.get(key)
        if record is None:
            raise ValueError(
                f"Missing record for routine={routine}, learning_rate={learning_rate}, lmbda={lmbda}."
            )
        records.append(record)
    return records[0], records[1]


def create_figures(
    *,
    raw_dirs: list[Path],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    retraction: str,
    pattern: str = "optimize_*_*.json",
    timestamp: str | None = None,
) -> list[Path]:
    """Create one 2xN loss/RMSE figure per learning rate."""
    records = successful_records(load_records(raw_dirs, pattern), retraction)
    if not records:
        raise FileNotFoundError(f"No successful constant-step records found under {raw_dirs}.")

    indexed = _index_records(records)
    learning_rates = sorted({_as_float(record, "learning_rate") for record in records})
    lmbdas = sorted({_as_float(record, "lmbda") for record in records}, reverse=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for learning_rate in learning_rates:
        fig, axes = plt.subplots(2, len(lmbdas), figsize=(5 * len(lmbdas), 7), constrained_layout=True)
        if len(lmbdas) == 1:
            axes = [[axes[0]], [axes[1]]]
        for column, lmbda in enumerate(lmbdas):
            reg_record, constraint_record = _require_pair(indexed, learning_rate=learning_rate, lmbda=lmbda)
            records_for_column = (reg_record, constraint_record)
            r_value = _as_float(constraint_record, "r")
            title = f"lambda={_format_float(lmbda)}, r={_format_float(r_value)}"

            for record in records_for_column:
                curve = _load_curve(record)
                label = str(record["routine"])
                iterations = range(1, len(curve["loss_history"]) + 1)
                axes[0][column].plot(iterations, curve["loss_history"], label=label, linewidth=1.6)
                axes[1][column].plot(iterations, curve["rmse_history"], label=label, linewidth=1.6)

            axes[0][column].set_title(title)
            axes[0][column].set_ylabel("Data loss")
            axes[1][column].set_ylabel("RMSE")
            axes[1][column].set_xlabel("Iteration")
            for row in range(2):
                axes[row][column].grid(True, alpha=0.25)
                axes[row][column].legend()

        lmbda_part = "_".join(_filename_float(value) for value in lmbdas)
        output = processed_dir / (
            "constant_step_optimize_reg_vs_optimize_constraint_constant_"
            f"{retraction}_lr{_filename_float(learning_rate)}_lmbdas{lmbda_part}_{stamp}.png"
        )
        fig.suptitle(f"Constant-step convergence, learning_rate={_format_float(learning_rate)}, retraction={retraction}")
        fig.savefig(output, dpi=200)
        plt.close(fig)
        outputs.append(output)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot constant-step optimizer grid loss/RMSE curves.")
    parser.add_argument("--raw-dir", type=Path, action="append", dest="raw_dirs", default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--retraction", choices=("qr", "polar"), required=True)
    parser.add_argument("--pattern", default="optimize_*_*.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dirs = args.raw_dirs or [DEFAULT_RAW_DIR]
    create_figures(
        raw_dirs=raw_dirs,
        processed_dir=args.processed_dir,
        retraction=args.retraction,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
