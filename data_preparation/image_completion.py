"""Build and validate finite image-completion artifacts."""

import torch


def _validate_mask_rate(mask_rate):
    if not 0.0 < mask_rate < 1.0:
        raise ValueError("mask_rate must be strictly between 0 and 1.")
    return float(mask_rate)


def prepare_image_completion(images, mask_rate, seed=42, meta=None):
    """Prepare a finite WLRA-compatible image-completion dataset."""
    if not images:
        raise ValueError("images must be nonempty.")
    mask_rate = _validate_mask_rate(mask_rate)

    flat = [image.flatten() for image in images]
    m_full = torch.stack(flat, dim=1).float()

    n_entries = m_full.numel()
    n_masked = int(round(n_entries * mask_rate))

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(n_entries, generator=generator)
    masked_flat_indices = permutation[:n_masked]

    w_flat = torch.ones(n_entries, dtype=torch.float32)
    w_flat[masked_flat_indices] = 0.0
    w = w_flat.reshape(m_full.shape)

    m_masked = m_full * w
    metadata = dict(meta or {})
    metadata.update(
        {
            "n_entries": n_entries,
            "n_masked": n_masked,
            "n_observed": n_entries - n_masked,
            "requested_mask_rate": mask_rate,
            "actual_mask_rate": n_masked / n_entries,
        }
    )

    data = {"M_full": m_full, "M_masked": m_masked, "W": w, "meta": metadata}
    validate_image_completion_dataset(data)
    return data


def prepare_nan_inspection_artifact(data):
    """Create a NaN-style companion artifact for local inspection."""
    validate_image_completion_dataset(data)
    m_full = data["M_full"]
    observed_mask = data["W"].bool()
    m_nan = m_full.clone()
    m_nan[~observed_mask] = float("nan")
    return {
        "M_full": m_full,
        "M_nan": m_nan,
        "observed_mask": observed_mask,
        "meta": dict(data.get("meta", {})),
    }


def validate_image_completion_dataset(data, expected_n=None, expected_rows=None):
    """Validate the finite artifact consumed by the experiment loaders."""
    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary.")
    missing = {"M_full", "M_masked", "W", "meta"} - set(data)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"data is missing required key(s): {names}.")

    m_full = data["M_full"]
    m_masked = data["M_masked"]
    w = data["W"]
    if not all(torch.is_tensor(tensor) for tensor in (m_full, m_masked, w)):
        raise ValueError("M_full, M_masked, and W must be torch tensors.")
    if m_full.ndim != 2:
        raise ValueError(f"M_full must be 2D, got ndim={m_full.ndim}.")
    if expected_rows is not None and m_full.shape[0] != expected_rows:
        raise ValueError(
            f"M_full must have {expected_rows} rows, got shape {tuple(m_full.shape)}."
        )
    if expected_n is not None and m_full.shape[1] != expected_n:
        raise ValueError(f"M_full must have {expected_n} columns, got {m_full.shape[1]}.")
    if m_full.shape != m_masked.shape or m_full.shape != w.shape:
        raise ValueError(
            "M_full, M_masked, and W must have the same shape. "
            f"Got {tuple(m_full.shape)}, {tuple(m_masked.shape)}, {tuple(w.shape)}."
        )
    if not torch.isfinite(m_full).all():
        raise ValueError("M_full must contain only finite values.")
    if not torch.isfinite(m_masked).all():
        raise ValueError("M_masked must contain only finite values.")
    if not torch.isfinite(w).all():
        raise ValueError("W must contain only finite values.")
    if not torch.logical_or(w == 0, w == 1).all():
        raise ValueError("W must be binary with entries equal to 0 or 1.")
    if not torch.equal(m_masked, m_full * w):
        raise ValueError("M_masked must equal M_full * W.")
    if not isinstance(data["meta"], dict):
        raise ValueError("meta must be a dictionary.")
