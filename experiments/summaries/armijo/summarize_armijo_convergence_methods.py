from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raws" / "armijo"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "results" / "processed" / "armijo"
DEFAULT_MANUSCRIPT_TABLE = REPO_ROOT / "paper" / "tables" / "full_train_20260618_armijo_grid_table.tex"
DEFAULT_RUN_PREFIX = "full_train_20260618"
DEFAULT_DATASET_FILENAME = "mnist0_n600_mask0.70_seed42_train.pt"
DEFAULT_PATTERNS = (
    "armijo_projection_arc_qr_summary_*.md",
    "armijo_feasible_direction_qr_summary_*.md",
    "armijo_wlra_reg_qr_summary_*.md",
)
NEW_SUMMARY_FILES = (
    "projection_arc_qr_summary.md",
    "feasible_direction_qr_summary.md",
    "regularized_qr_summary.md",
)
ROUTINE_DISPLAY = {
    "armijo_feasible_direction": "feasible-direction",
    "armijo_projection_arc": "projection-arc",
    "armijo_wlra_reg": "regularized-wlra",
}
ROUTINE_ORDER = tuple(ROUTINE_DISPLAY)
REGULARIZED_ARMIJO_DISPLAY = "Regularized Armijo (Benchmark)"
FEASIBLE_DIRECTION_DISPLAY = "Feasible Direction (Main Routine)"
PROJECTION_ARC_DISPLAY = "Projection Arc (Main Routine)"
MANUSCRIPT_ROUTINE_ORDER = (
    ("armijo_wlra_reg", REGULARIZED_ARMIJO_DISPLAY),
    ("armijo_feasible_direction", FEASIBLE_DIRECTION_DISPLAY),
    ("armijo_projection_arc", PROJECTION_ARC_DISPLAY),
)
MANUSCRIPT_RETRACTION_ORDER = ("qr", "polar")


@dataclass(frozen=True)
class MethodSummary:
    method: str
    source_file: Path
    records: int
    successful: int
    failed: int
    device: str
    average_runtime_seconds: float
    average_rmse: float
    average_loss: float
    dataset_files: tuple[str, ...]
    lmbdas: tuple[float, ...]
    rs: tuple[float, ...]
    max_backtracks: tuple[float, ...]
    best: dict[str, Any]
    worst: dict[str, Any]
    median: dict[str, Any]


@dataclass(frozen=True)
class ManuscriptGroupSummary:
    retraction: str
    rank: int
    lmbda: float
    routine: str
    best: dict[str, Any] | None
    mean_rmse: float | None
    mean_armijo_evaluations: float | None
    failures: int


def _strip_markdown(value: str) -> str:
    return value.strip().strip("*").strip()


def _parse_float(value: str) -> float | None:
    value = _strip_markdown(value)
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_duration_seconds(value: str) -> float | None:
    value = _strip_markdown(value)
    if value == "":
        return None
    match = re.fullmatch(r"(\d+)h\s+(\d+)m\s+(\d+)s", value)
    if match is None:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def _method_name(text: str, path: Path) -> str:
    first_line = text.splitlines()[0].strip("# ").strip()
    if "Projection-Arc" in first_line:
        return "projection-arc"
    if "Feasible-Direction" in first_line:
        return "feasible-direction"
    if "Regularized WLRA" in first_line:
        return "regularized-wlra"
    return path.stem


def _metadata_value(text: str, label: str) -> str:
    prefix = f"- {label}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _unique_floats(values: list[float | None]) -> tuple[float, ...]:
    unique = []
    for value in values:
        if value is None:
            continue
        if not any(math.isclose(value, item, rel_tol=1e-12, abs_tol=1e-12) for item in unique):
            unique.append(value)
    return tuple(unique)


def _unique_strings(values: list[str]) -> tuple[str, ...]:
    unique = []
    for value in values:
        value = value.strip()
        if value and value not in unique:
            unique.append(value)
    return tuple(unique)


def _parse_source_lmbda(text: str) -> float | None:
    source = _metadata_value(text, "Source")
    match = re.search(r"lmbda[-_]?([0-9.eE+-]+)", source)
    if match is None:
        return None
    return _parse_float(match.group(1).rstrip("_-/`"))


def _parse_dataset_files(text: str) -> tuple[str, ...]:
    for label in ("Dataset file", "Dataset"):
        value = _metadata_value(text, label).strip("`")
        if value:
            return _unique_strings([Path(value).name])
    return (DEFAULT_DATASET_FILENAME,)


def _format_float_list(values: tuple[float, ...]) -> str:
    if not values:
        return ""
    return ", ".join(_format_float(value) for value in values)


def _format_string_list(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _record_routine(record: dict[str, Any]) -> str:
    config = record.get("config")
    if isinstance(config, dict) and config.get("routine") is not None:
        return str(config["routine"])
    return str(record.get("routine", ""))


def _config_value(record: dict[str, Any], name: str) -> Any:
    config = record.get("config")
    if isinstance(config, dict) and name in config:
        return config[name]
    return record.get(name)


def _metric(record: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _record_loss(record: dict[str, Any]) -> float | None:
    """Return the fair unregularized WLRA loss for comparison summaries."""
    return _metric(record, "final_wlra_loss", "final_data_loss", "final_loss", "final_objective")


def load_json_records(raw_dir: Path, *, retraction: str | None = None) -> list[dict[str, Any]]:
    """Load raw Armijo JSON records, optionally filtering by retraction."""
    metadata_path = raw_dir / "grid_config.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    records = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name == "grid_config.json":
            continue
        record = json.loads(path.read_text())
        if retraction is not None and record.get("retraction") != retraction:
            continue
        record["_source_file"] = str(path)
        if metadata:
            record["_grid_config"] = metadata
        records.append(record)
    return records


def _group_key(record: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(record.get("retraction", "")),
        _record_routine(record),
        int(record.get("rank", 0)),
        float(record.get("lmbda", math.nan)),
    )


def _sorted_successes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [record for record in records if record.get("status") == "success"]
    return sorted(
        successful,
        key=lambda record: (
            _metric(record, "final_rmse") if _metric(record, "final_rmse") is not None else math.inf,
            _record_loss(record) if _record_loss(record) is not None else math.inf,
        ),
    )


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def load_prefixed_json_records(raw_root: Path = DEFAULT_RAW_DIR, *, run_prefix: str = DEFAULT_RUN_PREFIX) -> list[dict[str, Any]]:
    """Load raw Armijo records from run directories matching run_prefix."""
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    run_dirs = [
        path
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and path.name.startswith(run_prefix) and (path / "grid_config.json").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No raw Armijo run directories under {raw_root} matched prefix {run_prefix!r}.")
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        records.extend(load_json_records(run_dir))
    if not records:
        raise ValueError(f"No raw Armijo JSON records found under run directories matching {run_prefix!r}.")
    return records


def _record_rank(record: dict[str, Any]) -> int:
    return int(record.get("rank", 0))


def _record_lmbda(record: dict[str, Any]) -> float:
    return float(record.get("lmbda", math.nan))


def manuscript_group_summaries(records: list[dict[str, Any]]) -> dict[tuple[str, int, float, str], ManuscriptGroupSummary]:
    """Aggregate records for manuscript table rows by retraction, rank, lambda, and routine."""
    groups: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("retraction", "")), _record_rank(record), _record_lmbda(record), _record_routine(record))
        groups.setdefault(key, []).append(record)

    summaries = {}
    for (retraction, rank, lmbda, routine), group_records in groups.items():
        successes = _sorted_successes(group_records)
        summaries[(retraction, rank, lmbda, routine)] = ManuscriptGroupSummary(
            retraction=retraction,
            rank=rank,
            lmbda=lmbda,
            routine=routine,
            best=successes[0] if successes else None,
            mean_rmse=_mean([_metric(record, "final_rmse") for record in successes]),
            mean_armijo_evaluations=_mean([_metric(record, "mean_armijo_evaluations") for record in successes]),
            failures=len(group_records) - len(successes),
        )
    return summaries


def _latex_metric(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def _latex_mean_evaluations(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1f}"


def _latex_param_tuple(record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    beta = _format_float(_metric({"value": _config_value(record, "beta")}, "value"))
    sigma = _format_float(_metric({"value": _config_value(record, "sigma")}, "value"))
    step = _format_float(_metric({"value": _config_value(record, "initial_step")}, "value"))
    return rf"$({beta},{sigma},{step})$"


def _latex_rmse_with_params(record: dict[str, Any] | None) -> str:
    if record is None:
        return "--"
    rmse = _metric(record, "final_rmse")
    return rf"{_latex_metric(rmse)} {_latex_param_tuple(record)}"


def _bold_latex_cell(cell: str) -> str:
    if cell == "--":
        return cell
    return rf"\textbf{{{cell}}}"


def _latex_lmbda(value: float) -> str:
    if value > 0:
        exponent = round(math.log10(value))
        if math.isclose(value, 10**exponent, rel_tol=1e-12, abs_tol=1e-12):
            return rf"$10^{{{exponent}}}$"
    return _format_float(value)


def _routine_cells(
    summary: ManuscriptGroupSummary | None,
    *,
    is_winner: bool = False,
    include_armijo_evals: bool = False,
) -> list[str]:
    if summary is None:
        if include_armijo_evals:
            return ["--", "--", "--", "0"]
        return ["--", "--", "0"]
    best_cell = _latex_rmse_with_params(summary.best)
    if is_winner:
        best_cell = _bold_latex_cell(best_cell)
    cells = [
        best_cell,
        _latex_metric(summary.mean_rmse),
    ]
    if include_armijo_evals:
        cells.append(_latex_mean_evaluations(summary.mean_armijo_evaluations))
    cells.append(str(summary.failures))
    return cells


def _winner_routine(
    summaries: dict[tuple[str, int, float, str], ManuscriptGroupSummary],
    *,
    retraction: str,
    rank: int,
    lmbda: float,
) -> str | None:
    candidates = []
    for routine, _ in MANUSCRIPT_ROUTINE_ORDER:
        summary = summaries.get((retraction, rank, lmbda, routine))
        if summary is None or summary.best is None:
            continue
        rmse = _metric(summary.best, "final_rmse")
        loss = _record_loss(summary.best)
        if rmse is None:
            continue
        candidates.append((rmse, loss if loss is not None else math.inf, routine))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _rank_lambda_pairs(
    summaries: dict[tuple[str, int, float, str], ManuscriptGroupSummary],
    *,
    retraction: str,
) -> list[tuple[int, float]]:
    pairs = {(rank, lmbda) for item_retraction, rank, lmbda, _ in summaries if item_retraction == retraction}
    return sorted(pairs, key=lambda item: (item[0], -item[1]))


def _latex_routine_header(label: str) -> str:
    """Wrap manuscript routine labels without abbreviating their display names."""
    wrapped_labels = {
        REGULARIZED_ARMIJO_DISPLAY: r"\shortstack{Regularized Armijo\\(Benchmark)}",
        FEASIBLE_DIRECTION_DISPLAY: r"\shortstack{Feasible Direction\\(Main Routine)}",
        PROJECTION_ARC_DISPLAY: r"\shortstack{Projection Arc\\(Main Routine)}",
    }
    return wrapped_labels[label]


def manuscript_latex_table(records: list[dict[str, Any]], *, include_armijo_evals: bool = False) -> str:
    """Render a compact LaTeX tabular fragment for Armijo grid results."""
    summaries = manuscript_group_summaries(records)
    routine_columns = 4 if include_armijo_evals else 3
    total_columns = 2 + 3 * routine_columns
    column_spec = "cc|" + "|".join(["c" * routine_columns] * 3)
    routine_headers = []
    for index, (_, label) in enumerate(MANUSCRIPT_ROUTINE_ORDER):
        alignment = "c|" if index < len(MANUSCRIPT_ROUTINE_ORDER) - 1 else "c"
        routine_headers.append(
            rf"\multicolumn{{{routine_columns}}}{{{alignment}}}{{{_latex_routine_header(label)}}}"
        )
    routine_header = rf"rank & $\lambda$ & " + " & ".join(routine_headers) + r" \\"
    if include_armijo_evals:
        metric_cells = [
            r"\shortstack{best\\RMSE}",
            r"\shortstack{mean\\RMSE}",
            r"\shortstack{mean\\evals}",
            "failures",
        ]
    else:
        metric_cells = ["best RMSE", "mean RMSE", "failures"]
    metric_header = r" & & " + " & ".join(metric_cells * 3) + r" \\"
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
    ]
    for retraction_index, retraction in enumerate(MANUSCRIPT_RETRACTION_ORDER):
        if retraction_index > 0:
            lines.append(r"\hline")
        lines.extend(
            [
                rf"\multicolumn{{{total_columns}}}{{c}}{{\textbf{{{retraction.upper()} retraction}}}} \\",
                r"\hline",
                routine_header,
                metric_header,
                r"\hline",
            ]
        )
        previous_rank = None
        for rank, lmbda in _rank_lambda_pairs(summaries, retraction=retraction):
            if previous_rank is not None and rank != previous_rank:
                lines.append(r"\hline")
            cells = [str(rank), _latex_lmbda(lmbda)]
            winner_routine = _winner_routine(summaries, retraction=retraction, rank=rank, lmbda=lmbda)
            for routine, _ in MANUSCRIPT_ROUTINE_ORDER:
                cells.extend(
                    _routine_cells(
                        summaries.get((retraction, rank, lmbda, routine)),
                        is_winner=routine == winner_routine,
                        include_armijo_evals=include_armijo_evals,
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
            previous_rank = rank
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_manuscript_table(
    *,
    raw_root: Path = DEFAULT_RAW_DIR,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    output_tex: Path = DEFAULT_MANUSCRIPT_TABLE,
    include_armijo_evals: bool = False,
) -> Path:
    """Write an Armijo manuscript table as a LaTeX tabular fragment."""
    records = load_prefixed_json_records(raw_root, run_prefix=run_prefix)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(manuscript_latex_table(records, include_armijo_evals=include_armijo_evals))
    return output_tex


def _config_cells(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return ["", "", "", "", ""]
    return [
        _format_float(_metric(record, "final_rmse")),
        _format_float(_record_loss(record)),
        _format_float(_metric({"value": _config_value(record, "beta")}, "value")),
        _format_float(_metric({"value": _config_value(record, "sigma")}, "value")),
        _format_float(_metric({"value": _config_value(record, "initial_step")}, "value")),
    ]


def _main_summary_rows(records: list[dict[str, Any]]) -> list[list[str]]:
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    rows = []
    for retraction, routine, rank, lmbda in sorted(
        groups,
        key=lambda item: (item[0], ROUTINE_ORDER.index(item[1]) if item[1] in ROUTINE_ORDER else 99, item[2], item[3]),
    ):
        group_records = groups[(retraction, routine, rank, lmbda)]
        successes = _sorted_successes(group_records)
        best = successes[0] if successes else None
        median = successes[len(successes) // 2] if successes else None
        worst = successes[-1] if successes else None
        first = group_records[0]
        failures = len(group_records) - len(successes)
        rows.append(
            [
                retraction,
                ROUTINE_DISPLAY.get(routine, routine),
                str(rank),
                _format_float(lmbda),
                _format_float(_metric(first, "r")),
                *_config_cells(best),
                *_config_cells(median),
                *_config_cells(worst),
                _format_float(_mean([_metric(record, "final_rmse") for record in successes])),
                _format_float(_mean([_record_loss(record) for record in successes])),
                _format_float(_mean([_metric(record, "mean_armijo_evaluations") for record in successes])),
                str(len(successes)),
                str(failures),
            ]
        )
    return rows


def _failed_rows(records: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for record in records:
        if record.get("status") == "success":
            continue
        rows.append(
            [
                str(record.get("retraction", "")),
                ROUTINE_DISPLAY.get(_record_routine(record), _record_routine(record)),
                str(record.get("rank", "")),
                _format_float(_metric(record, "lmbda")),
                _format_float(_metric({"value": _config_value(record, "beta")}, "value")),
                _format_float(_metric({"value": _config_value(record, "sigma")}, "value")),
                _format_float(_metric({"value": _config_value(record, "initial_step")}, "value")),
                str(record.get("status", "")),
                str(record.get("error_type", "")),
                str(record.get("error", "")),
                str(record.get("_source_file", "")),
            ]
        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def markdown_json_summary(records: list[dict[str, Any]], *, raw_dir: Path, retraction: str | None = None) -> str:
    """Build grouped summary tables directly from raw JSON records."""
    if not records:
        raise ValueError(f"No Armijo JSON records found in {raw_dir}.")
    title = "Armijo Convergence Method Grid Summary"
    if retraction is not None:
        title = f"{retraction.upper()} Armijo Convergence Method Grid Summary"
    dataset_files = _unique_strings([str(record.get("_grid_config", {}).get("dataset_file", "")) for record in records])
    main_headers = [
        "retraction",
        "armijo_rule",
        "k",
        "lmbda",
        "r",
        "best final RMSE",
        "best final wlra loss",
        "best beta",
        "best sigma",
        "best initial_step",
        "median final RMSE",
        "median final wlra loss",
        "median beta",
        "median sigma",
        "median initial_step",
        "worst final RMSE",
        "worst final wlra loss",
        "worst beta",
        "worst sigma",
        "worst initial_step",
        "mean final RMSE",
        "mean final wlra loss",
        "mean armijo evals",
        "successes",
        "failures",
    ]
    failed_headers = [
        "retraction",
        "armijo_rule",
        "k",
        "lmbda",
        "beta",
        "sigma",
        "initial_step",
        "status",
        "error_type",
        "error",
        "source_file",
    ]
    lines = [
        f"# {title}",
        "",
        "- Best, median, and worst are selected by final RMSE, tie-broken by final unregularized WLRA loss.",
        "- Loss columns use `wlra()` for all methods; regularized objectives are diagnostics only.",
        f"- Source: `{raw_dir}`",
        f"- Records: {len(records)}",
    ]
    if dataset_files:
        lines.append(f"- Dataset file: {_format_string_list(dataset_files)}")
    retraction_names = [retraction] if retraction is not None else sorted({str(record.get("retraction", "")) for record in records})
    for name in retraction_names:
        section_records = [record for record in records if record.get("retraction") == name]
        lines.extend(["", f"## {str(name).upper()} Main Summary", "", *_markdown_table(main_headers, _main_summary_rows(section_records))])
    failed = _failed_rows(records)
    lines.extend(["", "## Failed Run Audit", ""])
    if failed:
        lines.extend(_markdown_table(failed_headers, failed))
    else:
        lines.append("No failed runs found.")
    lines.append("")
    return "\n".join(lines)


def parse_markdown_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Parse the first GitHub-flavored markdown table in path."""
    table_lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found in {path}.")
    headers = [_strip_markdown(cell) for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [_strip_markdown(cell) for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            raise ValueError(f"Malformed markdown table row in {path}: {line}")
        rows.append(dict(zip(headers, cells)))
    return headers, rows


def summarize_file(path: Path) -> MethodSummary:
    """Summarize one QR convergence-method markdown grid summary."""
    text = path.read_text()
    headers, rows = parse_markdown_table(path)
    step_column = "s" if "s" in headers else "s-bar"
    successes = []
    runtimes = []
    lmbdas = []
    rs = []
    max_backtracks = []
    for row in rows:
        rmse = _parse_float(row.get("final rmse", ""))
        loss = _parse_float(row.get("final loss", ""))
        runtime = _parse_duration_seconds(row.get("approx runtime", ""))
        if runtime is not None:
            runtimes.append(runtime)
        lmbdas.append(_parse_float(row.get("lambda", "")))
        rs.append(_parse_float(row.get("r", "")))
        max_backtracks.append(_parse_float(row.get("max-backtracks", "")))
        if rmse is None or loss is None:
            continue
        successes.append(
            {
                "beta": _parse_float(row.get("beta", "")),
                "sigma": _parse_float(row.get("sigma", "")),
                "step": _parse_float(row.get(step_column, "")),
                "step_label": step_column,
                "rmse": rmse,
                "loss": loss,
            }
        )

    if not successes:
        raise ValueError(f"No successful rows with finite RMSE/loss found in {path}.")
    by_rmse = sorted(successes, key=lambda row: (row["rmse"], row["loss"]))
    average_runtime = sum(runtimes) / len(runtimes) if runtimes else math.nan
    parsed_lmbdas = list(_unique_floats(lmbdas))
    source_lmbda = _parse_source_lmbda(text)
    if source_lmbda is not None and not any(
        math.isclose(source_lmbda, value, rel_tol=1e-12, abs_tol=1e-12) for value in parsed_lmbdas
    ):
        parsed_lmbdas.append(source_lmbda)
    return MethodSummary(
        method=_method_name(text, path),
        source_file=path,
        records=int(_metadata_value(text, "Records") or len(rows)),
        successful=len(successes),
        failed=len(rows) - len(successes),
        device=_metadata_value(text, "Device"),
        average_runtime_seconds=average_runtime,
        average_rmse=sum(row["rmse"] for row in successes) / len(successes),
        average_loss=sum(row["loss"] for row in successes) / len(successes),
        dataset_files=_parse_dataset_files(text),
        lmbdas=tuple(parsed_lmbdas),
        rs=_unique_floats(rs),
        max_backtracks=_unique_floats(max_backtracks),
        best=by_rmse[0],
        worst=by_rmse[-1],
        median=by_rmse[len(by_rmse) // 2],
    )


def latest_matching_file(processed_dir: Path, pattern: str) -> Path:
    """Return the newest markdown summary matching pattern."""
    matches = sorted(processed_dir.rglob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No files matched {processed_dir / pattern}.")


def load_method_summaries(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> list[MethodSummary]:
    """Load QR convergence-method summaries from a run folder or legacy tree."""
    new_files = [processed_dir / name for name in NEW_SUMMARY_FILES]
    if all(path.exists() for path in new_files):
        return [summarize_file(path) for path in new_files]
    return [summarize_file(latest_matching_file(processed_dir, pattern)) for pattern in DEFAULT_PATTERNS]


def _config_text(row: dict[str, Any]) -> str:
    return (
        f"beta={_format_float(row['beta'])}, "
        f"sigma={_format_float(row['sigma'])}, "
        f"{row['step_label']}={_format_float(row['step'])}"
    )


def markdown_summary(summaries: list[MethodSummary]) -> str:
    """Build a markdown comparison table across QR convergence methods."""
    dataset_files = _unique_strings([value for summary in summaries for value in summary.dataset_files])
    lmbdas = _unique_floats([value for summary in summaries for value in summary.lmbdas])
    rs = _unique_floats([value for summary in summaries for value in summary.rs])
    max_backtracks = _unique_floats([value for summary in summaries for value in summary.max_backtracks])
    headers = [
        "method",
        "records",
        "success/failed",
        "device",
        "avg runtime",
        "avg rmse",
        "avg loss",
        "best rmse/loss",
        "best config",
        "median rmse/loss",
        "median config",
        "worst rmse/loss",
        "worst config",
    ]
    rows = [
        "# QR Convergence Method Grid Summary",
        "",
        "- Best, median, and worst are selected by final RMSE, tie-broken by final loss.",
        "- Median uses the middle successful row after sorting by final RMSE; failed rows are excluded from numeric aggregates.",
        f"- Dataset file: {_format_string_list(dataset_files)}",
        f"- Lambda: {_format_float_list(lmbdas)}",
        f"- r: {_format_float_list(rs)}",
        f"- Max backtracks: {_format_float_list(max_backtracks)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for summary in summaries:
        cells = [
            summary.method,
            str(summary.records),
            f"{summary.successful}/{summary.failed}",
            summary.device,
            _format_duration(summary.average_runtime_seconds),
            _format_float(summary.average_rmse),
            _format_float(summary.average_loss),
            f"{_format_float(summary.best['rmse'])} / {_format_float(summary.best['loss'])}",
            _config_text(summary.best),
            f"{_format_float(summary.median['rmse'])} / {_format_float(summary.median['loss'])}",
            _config_text(summary.median),
            f"{_format_float(summary.worst['rmse'])} / {_format_float(summary.worst['loss'])}",
            _config_text(summary.worst),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    rows.extend(
        [
            "",
            "## Source Files",
            "",
            *[f"- {summary.method}: `{summary.source_file}`" for summary in summaries],
            "",
        ]
    )
    return "\n".join(rows)


def write_summary(
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    raw_dir: Path | None = None,
    retraction: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write a markdown summary for convergence-method grids."""
    del timestamp
    if raw_dir is not None:
        selected_retraction = None if retraction in {None, "all"} else retraction
        records = load_json_records(raw_dir, retraction=selected_retraction)
        processed_dir.mkdir(parents=True, exist_ok=True)
        suffix = selected_retraction if selected_retraction is not None else "all"
        output_path = processed_dir / f"convergence_methods_{suffix}_summary.md"
        output_path.write_text(markdown_json_summary(records, raw_dir=raw_dir, retraction=selected_retraction))
        return output_path
    summaries = load_method_summaries(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "convergence_methods_qr_summary.md"
    output_path.write_text(markdown_summary(summaries))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Armijo grid-search results across convergence methods.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--retraction", choices=("all", "qr", "polar"), default="all")
    parser.add_argument("--manuscript-table", action="store_true")
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_MANUSCRIPT_TABLE)
    parser.add_argument("--include-armijo-evals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manuscript_table:
        write_manuscript_table(
            raw_root=args.raw_root,
            run_prefix=args.run_prefix,
            output_tex=args.output_tex,
            include_armijo_evals=args.include_armijo_evals,
        )
        return
    write_summary(processed_dir=args.processed_dir, raw_dir=args.raw_dir, retraction=args.retraction)


if __name__ == "__main__":
    main()
