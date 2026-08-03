from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROUTINE_LABELS = {
    "armijo_feasible_direction": "feasible_direction",
    "armijo_projection_arc": "projection_arc",
    "armijo_wlra_reg": "regularized",
}
LABEL_ROUTINES = {value: key for key, value in ROUTINE_LABELS.items()}
RUN_ID_PATTERN = re.compile(r"^(?P<date>\d{6})-(?P<index>\d{2})$")


def allocate_run_id(base_dir: Path, *, now: datetime | None = None) -> str:
    """Return the next YYMMDD-NN run id for base_dir."""
    date_text = (now or datetime.now()).strftime("%y%m%d")
    max_index = 0
    if base_dir.exists():
        for path in base_dir.iterdir():
            if not path.is_dir():
                continue
            match = RUN_ID_PATTERN.fullmatch(path.name)
            if match is None or match.group("date") != date_text:
                continue
            max_index = max(max_index, int(match.group("index")))
    return f"{date_text}-{max_index + 1:02d}"


def routine_label(routine: str) -> str:
    """Return the stable folder label for an Armijo routine."""
    try:
        return ROUTINE_LABELS[routine]
    except KeyError as exc:
        raise ValueError(f"Unknown Armijo routine: {routine}.") from exc


def config_filename(*, routine: str, retraction: str, index: int) -> str:
    """Return a stable per-config JSON filename without grid parameters."""
    if index <= 0:
        raise ValueError("index must be positive.")
    label = routine_label(routine)
    return f"{label}_{retraction}_{index:03d}.json"


def config_path(raw_run_dir: Path, *, routine: str, retraction: str, index: int) -> Path:
    """Return the new-layout path for one Armijo config record."""
    label = routine_label(routine)
    return raw_run_dir / label / retraction / config_filename(routine=routine, retraction=retraction, index=index)


def grid_config_path(run_dir: Path) -> Path:
    """Return the shared grid-config metadata path for a raw or processed run directory."""
    return run_dir / "grid_config.json"


def write_grid_config(path: Path, config: dict[str, Any]) -> None:
    """Write shared grid-search metadata with deterministic JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

