"""Prepare grayscale CIFAR-10 data for the image-completion/WLRA task."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from .download import (
    load_cifar10_gray_train,
    load_cifar10_gray_train_from_parquet,
)
from .image_completion import (
    prepare_image_completion,
    prepare_nan_inspection_artifact,
)
from .sampling import (
    DEFAULT_CLASS_PERCENTAGES,
    collect_all,
    sample_by_class_percentages,
    validate_class_percentages,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = 10
DEFAULT_HF_TRAIN_PARQUET = (
    REPO_ROOT
    / "data"
    / "cifar10-hf"
    / "plain_text"
    / "train-00000-of-00001.parquet"
)


def parse_csv_floats(value):
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float value")
    if not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def parse_class_percentages(value):
    try:
        return validate_class_percentages(
            parse_csv_floats(value),
            n_classes=N_CLASSES,
            name="class_percentages",
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare grayscale CIFAR-10 data for manifold-grad-proj image "
            "completion."
        )
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
        help="Convert the full CIFAR-10 train split instead of sampling images.",
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
        "--class-percentages",
        type=parse_class_percentages,
        default=None,
        help=(
            "Comma-separated percentages for classes 0 through 9; values must "
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
        "--source",
        choices=("torchvision", "hf-parquet"),
        default="torchvision",
        help="CIFAR-10 source backend (default: torchvision).",
    )
    parser.add_argument(
        "--hf-train-parquet",
        type=Path,
        default=DEFAULT_HF_TRAIN_PARQUET,
        help=(
            "Path to a Hugging Face uoft-cs/cifar10 train parquet when "
            "--source=hf-parquet."
        ),
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


def _dataset_stem(n, mask_rate, seed, class_percentages):
    base = (
        f"cifar10-gray_classes0-9_n{n}_mask"
        f"{_format_mask_rate(mask_rate)}_seed{seed}"
    )
    if _is_balanced(class_percentages):
        return base

    payload = json.dumps(
        [round(value, 8) for value in class_percentages],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{base}_pct{digest}"


def _full_train_stem(mask_rate, seed):
    return f"cifar10-gray_train-full_mask{_format_mask_rate(mask_rate)}_seed{seed}"


def _validate_args(args):
    if args.full_train:
        if args.n is not None:
            raise ValueError("--n cannot be used with --full-train.")
        if args.class_percentages is not None:
            raise ValueError("--class-percentages cannot be used with --full-train.")
    elif args.n is not None and args.n <= 0:
        raise ValueError("--n must be positive.")
    for mask_rate in args.mask_rates:
        if not 0.0 < mask_rate < 1.0:
            raise ValueError("--mask-rates values must be strictly between 0 and 1.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")


def _sampled_args(args):
    n = args.n if args.n is not None else 1000
    class_percentages = (
        args.class_percentages
        if args.class_percentages is not None
        else DEFAULT_CLASS_PERCENTAGES
    )
    return n, class_percentages


def _resolve_source_path(path):
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_output_dir(path):
    return path if path.is_absolute() else REPO_ROOT / path


def _load_dataset(args):
    if args.source == "torchvision":
        return load_cifar10_gray_train(root=REPO_ROOT / "data")
    if args.source == "hf-parquet":
        return load_cifar10_gray_train_from_parquet(
            _resolve_source_path(args.hf_train_parquet)
        )
    raise ValueError(f"Unsupported source: {args.source!r}.")


def _dataset_source(dataset):
    path = getattr(dataset, "path", None)
    if path is not None:
        return str(path)
    return dataset.__class__.__name__


def _percentages_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        raise ValueError("class counts must contain at least one sample.")
    return tuple(100.0 * count / total for count in counts)


def _metadata(
    images,
    dataset,
    n_images,
    class_percentages,
    class_counts,
    seed,
    mask_rate,
    *,
    selection_mode="sampled",
):
    first = images[0]
    image_shape = list(first.shape[-2:])
    matrix_rows = int(first.numel())
    class_names = list(
        getattr(
            dataset,
            "classes",
            [str(class_id) for class_id in range(N_CLASSES)],
        )
    )
    return {
        "dataset": "CIFAR-10",
        "task": "wlra_image_completion",
        "split": "train",
        "train": True,
        "selection_mode": selection_mode,
        "source": _dataset_source(dataset),
        "transform": "grayscale_1_channel_to_tensor",
        "n_images": n_images,
        "class_percentages": list(class_percentages),
        "class_counts": list(class_counts),
        "class_names": class_names,
        "seed": seed,
        "mask_rate": mask_rate,
        "image_shape": image_shape,
        "channels": int(first.shape[0]) if first.ndim >= 3 else 1,
        "matrix_shape": [matrix_rows, n_images],
    }


def main(argv=None):
    args = parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CIFAR-10 train split as grayscale tensors from {args.source}...")
    dataset = _load_dataset(args)

    for seed in args.seeds:
        if args.full_train:
            print("Collecting all images from CIFAR-10 train split...")
            images, _labels, class_counts = collect_all(
                dataset,
                n_classes=N_CLASSES,
            )
            n_images = len(images)
            class_percentages = _percentages_from_counts(class_counts)
            selection_mode = "full_train"
        else:
            n_images, class_percentages = _sampled_args(args)
            print(
                f"Sampling {n_images} images from CIFAR-10 train split "
                f"(seed={seed})..."
            )
            images, _labels, class_counts = sample_by_class_percentages(
                dataset,
                n_images,
                class_percentages=class_percentages,
                seed=seed,
                n_classes=N_CLASSES,
            )
            selection_mode = "sampled"

        for mask_rate in args.mask_rates:
            meta = _metadata(
                images,
                dataset,
                n_images,
                class_percentages,
                class_counts,
                seed,
                mask_rate,
                selection_mode=selection_mode,
            )
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
                    class_percentages,
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
