from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "constant_step"
DEFAULT_PROCESSED_DIR = (
    REPO_ROOT / "results" / "processed" / "constant_step" / "optimize_reg_vs_optimize_constraint_constant" / "qr"
)
ROUTINES = ("optimize_reg", "optimize_constraint_constant")


def load_records(raw_dirs: list[Path], pattern: str = "optimize_*_*.json") -> list[dict[str, Any]]:
    """Load constant-step optimizer records from raw_dirs, excluding curve files."""
    records = []
    seen_paths = set()
    for raw_dir in raw_dirs:
        for path in sorted(raw_dir.rglob(pattern)):
            if "curves" in path.parts or path in seen_paths:
                continue
            seen_paths.add(path)
            record = json.loads(path.read_text())
            record["_source_file"] = str(path)
            records.append(record)
    return records


def _value(record: dict[str, Any], name: str) -> Any:
    config = record.get("config", {})
    if name in config:
        return config[name]
    return record.get(name)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.10g}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _format_unique_values(values: list[Any]) -> str:
    unique = []
    for value in values:
        if value is not None and value not in unique:
            unique.append(value)
    if not unique:
        return ""
    return ", ".join(str(value) for value in unique)


def _format_devices(records: list[dict[str, Any]]) -> str:
    return _format_unique_values([record.get("device") for record in records])


def _record_mtime(record: dict[str, Any]) -> datetime | None:
    source_file = record.get("_source_file")
    if source_file is None:
        return None
    return datetime.fromtimestamp(Path(source_file).stat().st_mtime)


def _run_start_time(record: dict[str, Any]) -> datetime | None:
    run_timestamp = record.get("run_timestamp")
    if run_timestamp is None:
        return None
    return datetime.strptime(str(run_timestamp), "%Y%m%d_%H%M%S")


def _runtime_seconds(record: dict[str, Any], previous_end_time: datetime | None) -> tuple[float | None, datetime | None]:
    end_time = _record_mtime(record)
    if end_time is None:
        return None, previous_end_time
    start_time = previous_end_time or _run_start_time(record)
    runtime = None if start_time is None else max(0.0, (end_time - start_time).total_seconds())
    return runtime, end_time


def total_running_time(records: list[dict[str, Any]]) -> float | None:
    """Estimate wall time from earliest run timestamp to latest record mtime."""
    starts = [_run_start_time(record) for record in records]
    ends = [_record_mtime(record) for record in records]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()


def best_success_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return successful record with lowest final RMSE, tie-broken by final loss."""
    successful = [record for record in records if record.get("status") == "success"]
    if not successful:
        return None
    return min(successful, key=lambda record: (record["final_rmse"], record["final_loss"]))


def _sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _value(record, "learning_rate") or 0,
        record.get("lmbda") or 0,
        record.get("routine", ""),
        record.get("run_timestamp", ""),
        record.get("config_index", 0),
    )


def _display_r(record: dict[str, Any]) -> Any:
    if record.get("routine") == "optimize_constraint_constant":
        return record.get("r")
    return None


def markdown_table(records: list[dict[str, Any]]) -> str:
    """Build one markdown table for all constant-step grid records."""
    best = best_success_record(records)
    headers = [
        "routine",
        "learning rate",
        "lambda",
        "r",
        "retraction",
        "status",
        "final rmse",
        "final loss",
        "best rmse",
        "best loss",
        "final objective",
        "final gradient norm",
        "final x norm",
        "approx runtime",
        "curve file",
        "error",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    previous_end_time = None
    for record in sorted(records, key=_sort_key):
        runtime, previous_end_time = _runtime_seconds(record, previous_end_time)
        cells = [
            str(record.get("routine", "")),
            _format_float(_value(record, "learning_rate")),
            _format_float(record.get("lmbda")),
            _format_float(_display_r(record)),
            str(record.get("retraction", "")),
            str(record.get("status", "")),
            _format_float(record.get("final_rmse")),
            _format_float(record.get("final_loss")),
            _format_float(record.get("best_rmse")),
            _format_float(record.get("best_loss")),
            _format_float(record.get("final_objective")),
            _format_float(record.get("final_gradient_norm")),
            _format_float(record.get("final_x_norm")),
            _format_duration(runtime),
            str(record.get("curve_file", "")),
            str(record.get("error", "")),
        ]
        if best is record:
            cells = [f"**{cell}**" for cell in cells]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_summary(
    *,
    raw_dirs: list[Path],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    retraction: str,
    pattern: str = "optimize_*_*.json",
    expected_records: int | None = 18,
    timestamp: str | None = None,
) -> Path:
    """Write a timestamped markdown summary for constant-step optimizer records."""
    records = [
        record
        for record in load_records(raw_dirs, pattern)
        if record.get("routine") in ROUTINES and record.get("retraction") == retraction
    ]
    if not records:
        raise FileNotFoundError(f"No constant-step records matched {raw_dirs}.")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(records)}.")

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / (
        "constant_step_optimize_reg_vs_optimize_constraint_constant_"
        f"{retraction}_summary_{stamp}.md"
    )
    best = best_success_record(records)
    lines = [
        "# Constant-Step Optimizer Grid Summary",
        "",
        f"- Source: `{', '.join(str(path) for path in raw_dirs)}`",
        f"- Records: {len(records)}",
        f"- Number of iterations: {_format_unique_values([record.get('iteration_numbers') for record in records])}",
        f"- Retraction: {retraction}",
        f"- Device: {_format_devices(records)}",
        f"- Total running time: {_format_duration(total_running_time(records))}",
        f"- Routines: {', '.join(ROUTINES)}",
        "- Best config is highlighted by lowest final RMSE, tie-broken by final loss.",
    ]
    if best is not None:
        lines.append(f"- Best final RMSE: {_format_float(best.get('final_rmse'))}")
    lines.extend(["", markdown_table(records), ""])
    output_path.write_text("\n".join(lines))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize constant-step optimizer grid JSON records as markdown.")
    parser.add_argument("--raw-dir", type=Path, action="append", dest="raw_dirs", default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--retraction", choices=("qr", "polar"), required=True)
    parser.add_argument("--pattern", default="optimize_*_*.json")
    parser.add_argument("--expected-records", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dirs = args.raw_dirs or [DEFAULT_RAW_DIR]
    expected_records = None if args.expected_records <= 0 else args.expected_records
    write_summary(
        raw_dirs=raw_dirs,
        processed_dir=args.processed_dir,
        retraction=args.retraction,
        pattern=args.pattern,
        expected_records=expected_records,
    )


if __name__ == "__main__":
    main()
