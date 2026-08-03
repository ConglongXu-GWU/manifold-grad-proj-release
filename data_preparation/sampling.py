"""Deterministic class-aware image sampling."""

import math

import torch


DEFAULT_DIGIT_PERCENTAGES = (10.0,) * 10
DEFAULT_CLASS_PERCENTAGES = (10.0,) * 10


def class_counts_from_labels(labels, n_classes=10):
    """Return per-class counts for integer labels in [0, n_classes)."""
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    counts = [0] * n_classes
    for label in labels:
        class_id = int(label)
        if not 0 <= class_id < n_classes:
            raise ValueError(
                f"Expected class label in [0, {n_classes - 1}], got {class_id}."
            )
        counts[class_id] += 1
    return tuple(counts)


def digit_counts_from_labels(labels):
    """Return per-digit counts for labels containing MNIST digits."""
    return class_counts_from_labels(labels, n_classes=10)


def collect_all(dataset, n_classes=10):
    """Collect every image and label in dataset order."""
    images, labels = [], []
    for index in range(len(dataset)):
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return images, labels, class_counts_from_labels(labels, n_classes=n_classes)


def sample(dataset, n, strategy="random", seed=42):
    """Sample images using the legacy random or stratified strategies."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    if strategy == "random":
        indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    elif strategy == "stratified":
        targets = dataset.targets
        classes = sorted(targets.unique().tolist())
        n_classes = len(classes)
        per_class = n // n_classes
        truncated_n = per_class * n_classes
        if truncated_n != n:
            print(
                f"[Stratified sampling] Requested {n} images truncated to {truncated_n} "
                f"(nearest multiple of {n_classes} classes, {per_class} per class)."
            )

        indices = []
        for class_id in classes:
            class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
            if per_class > len(class_indices):
                raise ValueError(
                    f"Requested {per_class} samples for class {class_id} but only "
                    f"{len(class_indices)} are available."
                )
            permutation = torch.randperm(len(class_indices), generator=generator)
            indices.extend(class_indices[permutation[:per_class]].tolist())

        permutation = torch.randperm(len(indices), generator=generator)
        indices = [indices[index] for index in permutation.tolist()]
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose 'random' or 'stratified'."
        )

    images, labels = [], []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return images, labels


def validate_digit_percentages(percentages):
    """Validate a 10-entry percentage vector for MNIST digits 0 through 9."""
    return validate_class_percentages(
        percentages,
        n_classes=10,
        name="digit_percentages",
    )


def validate_class_percentages(percentages, n_classes=10, name="class_percentages"):
    """Validate an n_classes-entry percentage vector."""
    if len(percentages) != n_classes:
        raise ValueError(f"{name} must contain exactly {n_classes} values.")

    values = tuple(float(value) for value in percentages)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} must be nonnegative.")
    if not math.isclose(sum(values), 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 100.")
    return values


def allocate_digit_counts(n, digit_percentages=DEFAULT_DIGIT_PERCENTAGES):
    """Allocate integer MNIST counts with largest remainders."""
    return allocate_class_counts(
        n,
        digit_percentages,
        n_classes=10,
        name="digit_percentages",
    )


def allocate_class_counts(
    n,
    class_percentages=DEFAULT_CLASS_PERCENTAGES,
    n_classes=10,
    name="class_percentages",
):
    """Allocate integer per-class counts summing exactly to n."""
    if n <= 0:
        raise ValueError("n must be positive.")
    percentages = validate_class_percentages(
        class_percentages,
        n_classes=n_classes,
        name=name,
    )

    raw_counts = [n * percentage / 100.0 for percentage in percentages]
    counts = [math.floor(value) for value in raw_counts]
    remaining = n - sum(counts)
    if remaining < 0:
        raise ValueError(f"{name} produced too many allocated samples.")

    order = sorted(
        range(n_classes),
        key=lambda class_id: (
            -(raw_counts[class_id] - counts[class_id]),
            class_id,
        ),
    )
    for class_id in order[:remaining]:
        counts[class_id] += 1

    if sum(counts) != n:
        raise ValueError("allocated class counts do not sum to n.")
    return tuple(counts)


def sample_by_digit_percentages(
    dataset,
    n,
    digit_percentages=DEFAULT_DIGIT_PERCENTAGES,
    seed=42,
):
    """Sample MNIST images with user-controlled digit percentages."""
    return sample_by_class_percentages(
        dataset,
        n,
        class_percentages=digit_percentages,
        seed=seed,
        n_classes=10,
        name="digit_percentages",
    )


def sample_by_class_percentages(
    dataset,
    n,
    class_percentages=DEFAULT_CLASS_PERCENTAGES,
    seed=42,
    n_classes=10,
    name="class_percentages",
):
    """Sample images with user-controlled class percentages."""
    counts = allocate_class_counts(
        n,
        class_percentages,
        n_classes=n_classes,
        name=name,
    )
    targets = torch.as_tensor(dataset.targets)

    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = []
    for class_id, count in enumerate(counts):
        if count == 0:
            continue
        class_indices = (targets == class_id).nonzero(as_tuple=True)[0]
        if count > len(class_indices):
            raise ValueError(
                f"Requested {count} samples for class {class_id}, but only "
                f"{len(class_indices)} are available."
            )
        permutation = torch.randperm(len(class_indices), generator=generator)
        indices.extend(class_indices[permutation[:count]].tolist())

    permutation = torch.randperm(len(indices), generator=generator)
    indices = [indices[index] for index in permutation.tolist()]

    images, labels = [], []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return images, labels, counts
