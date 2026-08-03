from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.runners.shared.armijo_layout import grid_config_path, routine_label, write_grid_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "results" / "raws" / "armijo"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "results" / "processed" / "armijo"


def load_grid_config(raw_dir: Path) -> dict[str, Any]:
    """Load shared run metadata when raw_dir is a new-layout run folder."""
    path = grid_config_path(raw_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _record_routine(record: dict[str, Any]) -> str | None:
    config = record.get("config")
    if isinstance(config, dict) and config.get("routine") is not None:
        return str(config["routine"])
    value = record.get("routine")
    return None if value is None else str(value)


def _record_retraction(record: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    value = record.get("retraction", metadata.get("retraction"))
    return None if value is None else str(value)


def load_records(
    raw_dir: Path,
    *,
    routine: str,
    retraction: str = "qr",
    pattern: str = "*.json",
) -> list[dict[str, Any]]:
    """Load Armijo records for one routine/retraction from old or new raw layouts."""
    metadata = load_grid_config(raw_dir)
    records = []
    for path in sorted(raw_dir.rglob(pattern)):
        if path.name == "grid_config.json":
            continue
        record = json.loads(path.read_text())
        if _record_routine(record) != routine:
            continue
        if _record_retraction(record, metadata) != retraction:
            continue
        record["_source_file"] = str(path)
        if metadata:
            record["_grid_config"] = metadata
        records.append(record)
    return records


def _value(record: dict[str, Any], name: str) -> Any:
    config = record.get("config", {})
    if isinstance(config, dict) and name in config:
        return config[name]
    return record.get(name)


def _metric(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None:
            return value
    return None


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.10g}"


def _format_bool(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _unique_values(records: list[dict[str, Any]], name: str) -> list[Any]:
    values = []
    for record in records:
        value = record.get(name)
        if value is not None and value not in values:
            values.append(value)
    return values


def _format_unique_values(values: list[Any]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return str(values[0])
    return ", ".join(str(value) for value in values)


def _format_device(value: Any) -> str:
    if value is None:
        return ""
    device = str(value)
    if device.startswith("mps"):
        return "gpu/mps"
    return device


def _format_devices(records: list[dict[str, Any]]) -> str:
    devices = []
    for record in records:
        device = _format_device(record.get("device"))
        if device and device not in devices:
            devices.append(device)
    return _format_unique_values(devices)


def total_running_time(records: list[dict[str, Any]]) -> float | None:
    """Estimate wall time from run timestamp to the latest checkpoint file mtime."""
    start_times = []
    end_times = []
    for record in records:
        run_timestamp = record.get("run_timestamp")
        source_file = record.get("_source_file")
        if run_timestamp is not None:
            start_times.append(datetime.strptime(str(run_timestamp), "%Y%m%d_%H%M%S"))
        if source_file is not None:
            end_times.append(datetime.fromtimestamp(Path(source_file).stat().st_mtime))
    if not start_times or not end_times:
        return None
    return (max(end_times) - min(start_times)).total_seconds()


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


def best_success_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the successful record with lowest final RMSE, tie-broken by loss."""
    successful = [record for record in records if record.get("status") == "success"]
    if not successful:
        return None
    return min(successful, key=lambda record: (record["final_rmse"], _metric(record, "final_loss", "final_data_loss")))


def markdown_table(records: list[dict[str, Any]], *, regularized: bool) -> str:
    """Build a markdown table for all loaded configs, highlighting the best row."""
    best = best_success_record(records)
    step_header = "s" if regularized else "s-bar"
    headers = [
        "beta",
        "sigma",
        step_header,
        "r",
        "lambda",
        "max-backtracks",
        "retraction",
        "final rmse",
        "final loss",
        "final data loss",
        "final gradient norm",
        "final x norm",
        "max-backtracks hit",
        "approx runtime",
        "final backtracks",
    ]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    sorted_records = sorted(
        records,
        key=lambda record: (
            record.get("run_timestamp", ""),
            record.get("config_index", 0),
            _value(record, "beta") or 0,
            _value(record, "sigma") or 0,
            _value(record, "initial_step") or 0,
        ),
    )
    previous_end_time = None
    for record in sorted_records:
        metadata = record.get("_grid_config", {})
        end_time = _record_mtime(record)
        if end_time is None:
            runtime = None
        else:
            start_time = previous_end_time or _run_start_time(record)
            runtime = None if start_time is None else max(0.0, (end_time - start_time).total_seconds())
            previous_end_time = end_time
        cells = [
            format_float(_value(record, "beta")),
            format_float(_value(record, "sigma")),
            format_float(_value(record, "initial_step")),
            format_float(record.get("r", metadata.get("r"))),
            format_float(record.get("lmbda", metadata.get("lmbda"))),
            format_float(_value(record, "max_backtracks")),
            str(record.get("retraction", metadata.get("retraction", ""))),
            format_float(record.get("final_rmse")),
            format_float(_metric(record, "final_loss", "final_data_loss", "final_objective")),
            format_float(record.get("final_data_loss")),
            format_float(record.get("final_gradient_norm")),
            format_float(record.get("final_x_norm")),
            _format_bool(record.get("max_backtracks_hit")),
            format_duration(runtime),
            format_float(record.get("final_backtracks")),
        ]
        if best is record:
            cells = [f"**{cell}**" if cell else "" for cell in cells]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _processed_run_dir(raw_dir: Path, processed_dir: Path) -> Path:
    if grid_config_path(raw_dir).exists() and processed_dir.name != raw_dir.name:
        return processed_dir / raw_dir.name
    return processed_dir


def write_summary(
    *,
    raw_dir: Path,
    processed_dir: Path,
    routine: str,
    retraction: str,
    title: str,
    output_name: str,
    regularized: bool,
    pattern: str = "*.json",
) -> Path:
    """Write a markdown summary for one Armijo routine/retraction grid."""
    records = load_records(raw_dir, routine=routine, retraction=retraction, pattern=pattern)
    if not records:
        raise FileNotFoundError(f"No JSON records for {routine}/{retraction} matched {raw_dir / pattern}.")
    metadata = load_grid_config(raw_dir)
    output_dir = _processed_run_dir(raw_dir, processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata:
        write_grid_config(grid_config_path(output_dir), metadata)
    output_path = output_dir / output_name
    best = best_success_record(records)
    iteration_numbers = _format_unique_values(_unique_values(records, "iteration_numbers"))
    running_time = format_duration(total_running_time(records))
    devices = _format_devices(records)
    lines = [
        f"# {title}",
        "",
        f"- Source: `{raw_dir / pattern}`",
        f"- Records: {len(records)}",
        f"- Dataset file: {metadata.get('dataset_file', '')}",
        f"- Lambda: {format_float(metadata.get('lmbda'))}",
        f"- r: {format_float(metadata.get('r'))}",
        f"- Number of iterations: {iteration_numbers}",
        f"- Total running time: {running_time}",
        f"- Device: {devices}",
        f"- Best config is highlighted by lowest final RMSE, tie-broken by final loss.",
    ]
    if best is not None:
        lines.append(f"- Best final RMSE: {format_float(best.get('final_rmse'))}")
    lines.extend(["", markdown_table(records, regularized=regularized), ""])
    output_path.write_text("\n".join(lines))
    return output_path


def build_parser(description: str, *, default_raw_dir: Path, default_processed_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir)
    parser.add_argument("--processed-dir", type=Path, default=default_processed_dir)
    parser.add_argument("--pattern", default="*.json")
    return parser
