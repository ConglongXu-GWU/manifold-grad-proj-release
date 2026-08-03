# Image-Completion Data Preparation

This package reproduces the MNIST and grayscale CIFAR-10 preprocessing used by
the image-completion experiments. Run all commands from the repository root.
The generators write PyTorch artifacts only when one of the commands below is
explicitly invoked.

## Environment

Use an environment separate from the core experiment environment:

```bash
python -m venv data_preparation/.venv
source data_preparation/.venv/bin/activate
python -m pip install -r data_preparation/requirements.txt
```

The dependency versions are pinned in `data_preparation/requirements.txt`.

## Artifact contract

Each finite artifact is a dictionary:

```python
{
    "M_full": torch.float32,    # (pixels, images)
    "M_masked": torch.float32,  # M_full * W
    "W": torch.float32,         # binary: 1 observed, 0 masked
    "meta": dict,
}
```

| Dataset | Image shape | Matrix shape |
|---|---:|---:|
| MNIST sample of `N` images | `1 x 28 x 28` | `(784, N)` |
| Grayscale CIFAR-10 sample of `N` images | `1 x 32 x 32` | `(1024, N)` |
| Full MNIST train split | 60,000 images | `(784, 60000)` |
| Full grayscale CIFAR-10 train split | 50,000 images | `(1024, 50000)` |

The same seed controls class sampling and mask generation. Class counts use
largest-remainder allocation, with ties resolved by class number.

## Existing sampled experiment datasets

Recreate the balanced MNIST-6000 artifacts for mask rates 0.25, 0.50, and
0.75:

```bash
python -m data_preparation.prepare_mnist \
  --n 6000 \
  --mask-rates 0.25,0.50,0.75 \
  --seeds 42 \
  --digit-percentages 10,10,10,10,10,10,10,10,10,10 \
  --output-dir data
```

The outputs have shape `(784, 6000)` and contain 600 images per digit.

Recreate the balanced grayscale CIFAR-10-5000 artifacts from the local Hugging
Face `uoft-cs/cifar10` parquet:

```bash
python -m data_preparation.prepare_cifar10 \
  --source hf-parquet \
  --hf-train-parquet data/cifar10-hf/plain_text/train-00000-of-00001.parquet \
  --n 5000 \
  --mask-rates 0.25,0.50,0.75 \
  --seeds 42 \
  --class-percentages 10,10,10,10,10,10,10,10,10,10 \
  --output-dir data
```

The outputs have shape `(1024, 5000)` and contain 500 images per class.
The same runner can download CIFAR-10 through torchvision by omitting
`--source hf-parquet`.

## Existing full-train variants

Recreate the full MNIST train artifact:

```bash
python -m data_preparation.prepare_mnist \
  --full-train \
  --mask-rates 0.50 \
  --seeds 42 \
  --output-dir data
```

Recreate the full grayscale CIFAR-10 train artifact:

```bash
python -m data_preparation.prepare_cifar10 \
  --source hf-parquet \
  --hf-train-parquet data/cifar10-hf/plain_text/train-00000-of-00001.parquet \
  --full-train \
  --mask-rates 0.50 \
  --seeds 42 \
  --output-dir data
```

Full-train artifacts remain ignored because they exceed normal GitHub file-size
limits. Sampled artifact tracking is unchanged.

## Additional controls

Both entry points accept comma-separated seeds and mask rates. Sampled mode
accepts custom ten-class percentages, and `--save-nan-artifact` creates a local
inspection companion. Use `--help` for the complete command surface.

The CIFAR metadata preserves the source behavior of storing the resolved local
parquet path. Compare regenerated CIFAR artifacts by tensor contents and
semantic metadata rather than requiring an identical serialized file hash
across machines.
