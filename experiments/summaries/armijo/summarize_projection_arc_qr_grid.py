from __future__ import annotations

import json
from pathlib import Path

from experiments.summaries.armijo.grid_summary_common import (
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    best_success_record,
    build_parser,
    load_records as _load_records,
    markdown_table,
    write_summary as _write_summary,
)


ROUTINE = "armijo_projection_arc"
RETRACTION = "qr"
DEFAULT_RAW_DIR = DEFAULT_RAW_ROOT
DEFAULT_PROCESSED_DIR = DEFAULT_PROCESSED_ROOT
DEFAULT_PATTERN = "*.json"
OUTPUT_NAME = "projection_arc_qr_summary.md"


def load_records(raw_dir: Path, pattern: str = DEFAULT_PATTERN):
    """Load projection-arc QR grid-search JSON records from old or new raw layouts."""
    return _load_records(raw_dir, routine=ROUTINE, retraction=RETRACTION, pattern=pattern)


def write_summary(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    pattern: str = DEFAULT_PATTERN,
    timestamp: str | None = None,
) -> Path:
    """Write a markdown summary for projection-arc QR raw JSON files."""
    del timestamp
    return _write_summary(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        routine=ROUTINE,
        retraction=RETRACTION,
        title="Projection-Arc QR Grid Search Summary",
        output_name=OUTPUT_NAME,
        regularized=False,
        pattern=pattern,
    )


def parse_args():
    return build_parser(
        "Summarize projection-arc QR grid JSON records as markdown.",
        default_raw_dir=DEFAULT_RAW_DIR,
        default_processed_dir=DEFAULT_PROCESSED_DIR,
    ).parse_args()


def main() -> None:
    args = parse_args()
    write_summary(raw_dir=args.raw_dir, processed_dir=args.processed_dir, pattern=args.pattern)


if __name__ == "__main__":
    main()
