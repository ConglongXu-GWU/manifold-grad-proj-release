"""Public-dataset loaders used by the preprocessing entry points."""

from io import BytesIO
from pathlib import Path

import torch


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Dataset preparation requires torchvision. Install "
            "data_preparation/requirements.txt."
        ) from exc
    return datasets, transforms


def _require_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR-10 parquet loading requires Pillow. Install "
            "data_preparation/requirements.txt."
        ) from exc
    return Image


def _read_parquet(path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR-10 parquet loading requires pandas and pyarrow. Install "
            "data_preparation/requirements.txt."
        ) from exc
    return pd.read_parquet(path)


def _cifar_gray_transform():
    _, transforms = _require_torchvision()
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


def load_mnist_train(root="data"):
    datasets, transforms = _require_torchvision()
    return datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )


def load_cifar10_gray_train(root="data"):
    datasets, _ = _require_torchvision()
    return datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=_cifar_gray_transform(),
    )


class Cifar10GrayParquet:
    """Torchvision-style view over the Hugging Face CIFAR-10 train parquet."""

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.frame = _read_parquet(self.path)
        missing = {"img", "label"} - set(self.frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"CIFAR-10 parquet is missing column(s): {names}.")
        self.targets = torch.as_tensor(
            self.frame["label"].to_list(),
            dtype=torch.long,
        )
        self.transform = _cifar_gray_transform()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = _image_from_parquet_value(row["img"])
        return self.transform(image), int(row["label"])


def _image_from_parquet_value(value):
    Image = _require_pillow_image()
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is None:
            path = value.get("path")
            if path is None:
                raise ValueError("Image dictionary must contain 'bytes' or 'path'.")
            return Image.open(path).convert("RGB")
        return Image.open(BytesIO(raw)).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Unsupported parquet image value type: {type(value)!r}.")


def load_cifar10_gray_train_from_parquet(path):
    return Cifar10GrayParquet(path)
