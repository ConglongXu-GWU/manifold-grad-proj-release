from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.shared.armijo_grid import parse_csv_floats, parse_csv_ints, parse_csv_retractions
from experiments.plotting.shared.convergence_common import (
    create_grid_convergence_figures,
    load_curve_records,
    plot_merged_grid_convergence_curves,
)


def parse_csv_metrics(value: str) -> list[str]:
    metrics = [part.strip() for part in value.split(",") if part.strip()]
    if not metrics:
        raise ValueError("Expected at least one metric.")
    unknown = sorted(set(metrics) - {"loss", "rmse"})
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot 3x3 Armijo convergence grids from history-mode raw records.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--raw-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "figures" / "armijo" / "convergence_methods")
    parser.add_argument("--retractions", type=parse_csv_retractions, default=["qr", "polar"])
    parser.add_argument("--ranks", type=parse_csv_ints, default=[32, 64, 128])
    parser.add_argument("--lmbdas", type=parse_csv_floats, default=[1e-2, 1e-4, 1e-6])
    parser.add_argument("--metrics", type=parse_csv_metrics, default=["loss", "rmse"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs="+",
        default=None,
        metavar="LABEL_OR_RAW_DIR",
        help="Merged mode: repeat as --dataset LABEL RAW_DIR [RAW_DIR ...].",
    )
    parser.add_argument("--merged-output", type=Path, default=None)
    parser.add_argument(
        "--merged-font-size",
        type=float,
        default=34.0,
        help="Source font size for merged figures (34 pt renders near 9 pt at manuscript full width).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset is not None:
        if args.raw_dir is not None or args.raw_dirs is not None:
            raise SystemExit(
                "Merged --dataset mode cannot be combined with --raw-dir or --raw-dirs."
            )
        if args.merged_output is None:
            raise SystemExit("--merged-output is required with --dataset.")
        if len(args.metrics) != 1:
            raise SystemExit("Merged --dataset mode requires exactly one metric.")
        datasets = []
        for dataset in args.dataset:
            if len(dataset) < 2:
                raise SystemExit(
                    "Each --dataset requires a label followed by at least one raw directory."
                )
            label, *raw_dirs = dataset
            datasets.append(
                (label, load_curve_records([Path(raw_dir) for raw_dir in raw_dirs]))
            )
        plot_merged_grid_convergence_curves(
            datasets,
            retractions=args.retractions,
            metric=args.metrics[0],
            output_path=args.merged_output,
            ranks=args.ranks,
            lmbdas=args.lmbdas,
            font_size=args.merged_font_size,
        )
        return

    raw_dirs = []
    if args.raw_dir is not None:
        raw_dirs.append(args.raw_dir)
    if args.raw_dirs is not None:
        raw_dirs.extend(args.raw_dirs)
    if not raw_dirs:
        raise SystemExit("At least one of --raw-dir or --raw-dirs is required.")
    create_grid_convergence_figures(
        raw_dirs=raw_dirs,
        output_dir=args.output_dir,
        retractions=args.retractions,
        ranks=args.ranks,
        lmbdas=args.lmbdas,
        metrics=args.metrics,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
