"""Prepare MNIST data for the image-completion/WLRA task."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from .download import load_mnist_train
from .image_completion import (
    prepare_image_completion,
    prepare_nan_inspection_artifact,
)
from .sampling import (
    DEFAULT_DIGIT_PERCENTAGES,
    collect_all,
    sample_by_digit_percentages,
    validate_digit_percentages,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_csv_floats(value):
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float value")
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def parse_digit_percentages(value):
    try:
        return validate_digit_percentages(parse_csv_floats(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser():
    parser = argparse.ArgumentParser(
        description="Prepare MNIST data for manifold-grad-proj image completion."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of train images to sample (default: 1000).",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Convert the full MNIST train split instead of sampling images.",
    )
    parser.add_argument(
        "--mask-rates",
        type=parse_csv_floats,
        default=[0.50],
        help="Comma-separated mask rates in (0, 1) (default: 0.50).",
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: [
            int(part.strip()) for part in value.split(",") if part.strip()
        ],
        default=[42],
        help="Comma-separated random seeds (default: 42).",
    )
    parser.add_argument(
        "--digit-percentages",
        type=parse_digit_percentages,
        default=None,
        help=(
            "Comma-separated percentages for digits 0 through 9; values must "
            "sum to 100 (default: balanced 10%% each)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for finite artifacts (default: data).",
    )
    parser.add_argument(
        "--save-nan-artifact",
        action="store_true",
        help="Also save local NaN-style inspection artifacts.",
    )
    return parser


def parse_args(argv=None):
    return _parser().parse_args(argv)


def _format_mask_rate(mask_rate):
    return f"{mask_rate:.2f}"


def _is_balanced(percentages):
    return all(
        math.isclose(value, 10.0, rel_tol=0.0, abs_tol=1e-6)
        for value in percentages
    )


def _dataset_stem(n, mask_rate, seed, digit_percentages):
    base = f"mnist_digits0-9_n{n}_mask{_format_mask_rate(mask_rate)}_seed{seed}"
    if _is_balanced(digit_percentages):
        return base

    payload = json.dumps(
        [round(value, 8) for value in digit_percentages],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{base}_pct{digest}"


def _full_train_stem(mask_rate, seed):
    return f"mnist_train-full_mask{_format_mask_rate(mask_rate)}_seed{seed}"


def _percentages_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        raise ValueError("digit counts must contain at least one sample.")
    return tuple(100.0 * count / total for count in counts)


def _validate_args(args):
    if args.full_train:
        if args.n is not None:
            raise ValueError("--n cannot be used with --full-train.")
        if args.digit_percentages is not None:
            raise ValueError("--digit-percentages cannot be used with --full-train.")
    elif args.n is not None and args.n <= 0:
        raise ValueError("--n must be positive.")
    for mask_rate in args.mask_rates:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError("--mask-rates values must be strictly between 0 and 1.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")


def _sampled_args(args):
    n = args.n if args.n is not None else 1000
    digit_percentages = (
        args.digit_percentages
        if args.digit_percentages is not None
        else DEFAULT_DIGIT_PERCENTAGES
    )
    return n, digit_percentages


def _resolve_output_dir(path):
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv=None):
    args = parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST train split...")
    dataset = load_mnist_train(root=REPO_ROOT / "data")

    for seed in args.seeds:
        if args.full_train:
            print("Collecting all images from MNIST train split...")
            images, _labels, digit_counts = collect_all(dataset)
            n_images = len(images)
            digit_percentages = _percentages_from_counts(digit_counts)
            selection_mode = "full_train"
        else:
            n_images, digit_percentages = _sampled_args(args)
            print(f"Sampling {n_images} images from MNIST train split (seed={seed})...")
            images, _labels, digit_counts = sample_by_digit_percentages(
                dataset,
                n_images,
                digit_percentages=digit_percentages,
                seed=seed,
            )
            selection_mode = "sampled"

        for mask_rate in args.mask_rates:
            meta = {
                "dataset": "MNIST",
                "task": "wlra_image_completion",
                "split": "train",
                "train": True,
                "selection_mode": selection_mode,
                "n_images": n_images,
                "digit_percentages": list(digit_percentages),
                "digit_counts": list(digit_counts),
                "seed": seed,
                "mask_rate": mask_rate,
                "image_shape": [28, 28],
                "matrix_shape": [784, n_images],
            }
            print(f"Preparing mask_rate={mask_rate:.2f} for seed={seed}...")
            data = prepare_image_completion(
                images,
                mask_rate=mask_rate,
                seed=seed,
                meta=meta,
            )

            stem = (
                _full_train_stem(mask_rate, seed)
                if args.full_train
                else _dataset_stem(
                    n_images,
                    mask_rate,
                    seed,
                    digit_percentages,
                )
            )
            local_path = output_dir / f"{stem}.pt"
            torch.save(data, local_path)
            print(f"Saved finite artifact: {local_path}")

            if args.save_nan_artifact:
                nan_path = output_dir / f"{stem}_nan.pt"
                torch.save(prepare_nan_inspection_artifact(data), nan_path)
                print(f"Saved NaN inspection artifact: {nan_path}")

            n_total = data["W"].numel()
            n_observed = int(data["W"].sum().item())
            n_masked = n_total - n_observed
            print(f"  Matrix shape : {tuple(data['M_full'].shape)}  (pixels x images)")
            print(f"  Observed     : {n_observed}  ({100 * n_observed / n_total:.1f}%)")
            print(f"  Masked       : {n_masked}  ({100 * n_masked / n_total:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
