from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from data_preparation import download
from data_preparation import prepare_cifar10
from data_preparation import prepare_mnist
from data_preparation.image_completion import (
    prepare_image_completion,
    prepare_nan_inspection_artifact,
    validate_image_completion_dataset,
)
from data_preparation.sampling import (
    allocate_class_counts,
    allocate_digit_counts,
    collect_all,
    sample_by_class_percentages,
    sample_by_digit_percentages,
)


class FakeImageDataset:
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

    def __init__(self, counts, image_size):
        self.targets = [
            class_id
            for class_id, count in enumerate(counts)
            for _ in range(count)
        ]
        self.images = [
            torch.full(
                (1, image_size, image_size),
                fill_value=index / max(len(self.targets), 1),
            )
            for index in range(len(self.targets))
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


def test_largest_remainder_allocation_and_balanced_defaults():
    assert allocate_digit_counts(1000) == (100,) * 10
    assert allocate_class_counts(1000) == (100,) * 10
    assert allocate_digit_counts(
        7,
        [50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
    ) == (4, 3, 0, 0, 0, 0, 0, 0, 0, 0)


def test_class_sampling_is_balanced_and_deterministic():
    dataset = FakeImageDataset([20] * 10, image_size=28)

    first_images, first_labels, first_counts = sample_by_digit_percentages(
        dataset,
        20,
        seed=123,
    )
    second_images, second_labels, second_counts = sample_by_class_percentages(
        dataset,
        20,
        seed=123,
    )

    assert first_counts == second_counts == (2,) * 10
    assert first_labels == second_labels
    assert torch.equal(torch.stack(first_images), torch.stack(second_images))


def test_collect_all_preserves_order_and_counts():
    dataset = FakeImageDataset([2, 1, 0, 0, 0, 0, 0, 0, 0, 1], image_size=28)

    images, labels, counts = collect_all(dataset)

    assert len(images) == 4
    assert labels == [0, 0, 1, 9]
    assert counts == (2, 1, 0, 0, 0, 0, 0, 0, 0, 1)


@pytest.mark.parametrize(
    ("image_size", "expected_rows"),
    [(28, 784), (32, 1024)],
)
def test_image_completion_schema_and_shape(image_size, expected_rows):
    images = [torch.rand(1, image_size, image_size) for _ in range(5)]

    data = prepare_image_completion(images, mask_rate=0.25, seed=42)

    assert sorted(data) == ["M_full", "M_masked", "W", "meta"]
    assert data["M_full"].shape == (expected_rows, 5)
    assert data["M_masked"].shape == (expected_rows, 5)
    assert data["W"].shape == (expected_rows, 5)
    assert torch.logical_or(data["W"] == 0, data["W"] == 1).all()
    assert torch.equal(data["M_masked"], data["M_full"] * data["W"])
    assert data["meta"]["n_masked"] == round(expected_rows * 5 * 0.25)
    validate_image_completion_dataset(
        data,
        expected_n=5,
        expected_rows=expected_rows,
    )


def test_image_completion_mask_is_deterministic_by_seed():
    images = [torch.rand(1, 28, 28) for _ in range(3)]

    first = prepare_image_completion(images, mask_rate=0.75, seed=7)
    second = prepare_image_completion(images, mask_rate=0.75, seed=7)

    assert torch.equal(first["W"], second["W"])
    assert torch.equal(first["M_masked"], second["M_masked"])


def test_nan_companion_uses_the_finite_artifact_mask():
    data = prepare_image_completion(
        [torch.ones(1, 28, 28)],
        mask_rate=0.5,
        seed=42,
    )

    companion = prepare_nan_inspection_artifact(data)

    assert torch.isnan(
        companion["M_nan"][~companion["observed_mask"]]
    ).all()
    assert torch.equal(
        companion["M_nan"][companion["observed_mask"]],
        companion["M_full"][companion["observed_mask"]],
    )


def test_balanced_output_filenames_match_existing_artifacts():
    assert (
        prepare_mnist._dataset_stem(6000, 0.75, 42, [10] * 10)
        == "mnist_digits0-9_n6000_mask0.75_seed42"
    )
    assert (
        prepare_mnist._full_train_stem(0.5, 42)
        == "mnist_train-full_mask0.50_seed42"
    )
    assert (
        prepare_cifar10._dataset_stem(5000, 0.25, 42, [10] * 10)
        == "cifar10-gray_classes0-9_n5000_mask0.25_seed42"
    )
    assert (
        prepare_cifar10._full_train_stem(0.5, 42)
        == "cifar10-gray_train-full_mask0.50_seed42"
    )


def test_mnist_full_train_argument_validation():
    args = SimpleNamespace(
        full_train=True,
        n=6000,
        digit_percentages=None,
        mask_rates=[0.5],
        seeds=[42],
    )

    with pytest.raises(ValueError, match="--n cannot be used"):
        prepare_mnist._validate_args(args)


def test_cifar_full_train_argument_validation():
    args = SimpleNamespace(
        full_train=True,
        n=None,
        class_percentages=[10] * 10,
        mask_rates=[0.5],
        seeds=[42],
    )

    with pytest.raises(ValueError, match="--class-percentages cannot be used"):
        prepare_cifar10._validate_args(args)


def test_cifar_metadata_records_sampled_and_full_train_shapes():
    dataset = FakeImageDataset([2] * 10, image_size=32)
    sampled_images, _, sampled_counts = sample_by_class_percentages(
        dataset,
        10,
        seed=42,
    )

    sampled = prepare_cifar10._metadata(
        sampled_images,
        dataset,
        10,
        [10] * 10,
        sampled_counts,
        seed=42,
        mask_rate=0.5,
    )
    full_train = prepare_cifar10._metadata(
        [sampled_images[0]],
        dataset,
        50000,
        [10] * 10,
        [5000] * 10,
        seed=42,
        mask_rate=0.5,
        selection_mode="full_train",
    )

    assert sampled["matrix_shape"] == [1024, 10]
    assert sampled["class_counts"] == [1] * 10
    assert sampled["class_names"] == FakeImageDataset.classes
    assert full_train["selection_mode"] == "full_train"
    assert full_train["matrix_shape"] == [1024, 50000]
    assert full_train["class_counts"] == [5000] * 10


def test_cifar_source_selection(monkeypatch):
    torchvision_dataset = object()
    parquet_dataset = object()
    monkeypatch.setattr(
        prepare_cifar10,
        "load_cifar10_gray_train",
        lambda root: torchvision_dataset,
    )
    monkeypatch.setattr(
        prepare_cifar10,
        "load_cifar10_gray_train_from_parquet",
        lambda path: parquet_dataset,
    )

    assert (
        prepare_cifar10._load_dataset(
            SimpleNamespace(
                source="torchvision",
                hf_train_parquet=None,
            )
        )
        is torchvision_dataset
    )
    assert (
        prepare_cifar10._load_dataset(
            SimpleNamespace(
                source="hf-parquet",
                hf_train_parquet=prepare_cifar10.DEFAULT_HF_TRAIN_PARQUET,
            )
        )
        is parquet_dataset
    )


def test_hf_parquet_adapter_uses_mocked_in_memory_frame(monkeypatch):
    pandas = pytest.importorskip("pandas")
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("torchvision")

    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    frame = pandas.DataFrame(
        {
            "img": [{"bytes": buffer.getvalue(), "path": None}],
            "label": [3],
        }
    )
    monkeypatch.setattr(download, "_read_parquet", lambda _path: frame)

    dataset = download.load_cifar10_gray_train_from_parquet(
        "unused-train.parquet"
    )
    item, label = dataset[0]

    assert len(dataset) == 1
    assert dataset.targets.tolist() == [3]
    assert item.shape == (1, 32, 32)
    assert item.dtype == torch.float32
    assert label == 3
