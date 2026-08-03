# Retraction-Based Armijo Methods for Matrix Completion

This repository contains a compact implementation of three Armijo routines for
weighted low-rank matrix completion on a product of two Stiefel manifolds and a
Euclidean factor:

- **Feasible Direction** projects one descent direction and then backtracks on
  its scalar multiplier.
- **Projection Arc** recomputes the feasible projection at every backtracking
  step.
- **Regularized Armijo** applies sufficient decrease to a regularized weighted
  low-rank objective.

Both QR and polar retractions are supported. Small processed CSV files and PNG
figures under `results/` provide reference outputs. Dataset binaries and raw run
records are intentionally not distributed.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r data_preparation/requirements.txt
```

The package can be imported directly from the source tree by setting
`PYTHONPATH=src`, as in the commands below.

## Prepare the datasets

Users must download MNIST and CIFAR-10 and run the preprocessing commands before
reproducing experiments. The preprocessing package creates matrices with
columns representing grayscale images and stores `M_full`, `M_masked`, and the
binary observation mask `W` in a local PyTorch artifact.

```bash
python -m data_preparation.prepare_mnist \
  --n 6000 \
  --mask-rates 0.25,0.50,0.75 \
  --seeds 42 \
  --digit-percentages 10,10,10,10,10,10,10,10,10,10 \
  --output-dir data

python -m data_preparation.prepare_cifar10 \
  --n 5000 \
  --mask-rates 0.25,0.50,0.75 \
  --seeds 42 \
  --class-percentages 10,10,10,10,10,10,10,10,10,10 \
  --output-dir data
```

See `data_preparation/README.md` for the artifact schema and optional controls.
The sampled MNIST matrix has shape `(784, 6000)`, while the sampled grayscale
CIFAR-10 matrix has shape `(1024, 5000)`.

## Run QR and polar grids

The combined grid runners accept all three routines. The example below uses one
value per parameter; expand the comma-separated lists for a full grid.

```bash
PYTHONPATH=src python experiments/runners/armijo/grid_search_qr.py \
  --dataset-path data/mnist_digits0-9_n6000_mask0.50_seed42.pt \
  --run-name mnist_qr_grid \
  --routines all \
  --betas 0.5 \
  --sigmas 0.25 \
  --initial-steps 0.3 \
  --max-backtracks 200 \
  --ranks 32,64,128 \
  --lmbdas 1e-2,1e-4,1e-6 \
  --iterations 200 \
  --r auto \
  --rmse-mode final \
  --device cpu \
  --dtype float32

PYTHONPATH=src python experiments/runners/armijo/grid_search_polar.py \
  --dataset-path data/mnist_digits0-9_n6000_mask0.50_seed42.pt \
  --run-name mnist_polar_grid \
  --routines all \
  --betas 0.5 \
  --sigmas 0.25 \
  --initial-steps 0.3 \
  --max-backtracks 200 \
  --ranks 32,64,128 \
  --lmbdas 1e-2,1e-4,1e-6 \
  --iterations 200 \
  --r auto \
  --rmse-mode final \
  --device cpu \
  --dtype float32
```

Each runner checkpoints raw records under `results/raws/armijo/` and writes
best-configuration summaries under `results/processed/armijo/`. Raw outputs are
local reproducibility products and are ignored by Git.

## Long-run histories and plots

Use the best configurations from a completed grid to generate iteration
histories. Run the command once for each routine and retraction.

```bash
PYTHONPATH=src python experiments/runners/armijo/run_selected_history_grid.py \
  --dataset-path data/mnist_digits0-9_n6000_mask0.50_seed42.pt \
  --source-run-prefix mnist \
  --run-name mnist_qr_history \
  --routine armijo_feasible_direction \
  --retraction qr \
  --ranks 32,64,128 \
  --lmbdas 1e-2,1e-4,1e-6 \
  --iterations 1000 \
  --r auto \
  --device cpu \
  --dtype float32

PYTHONPATH=src python experiments/plotting/armijo/curves/plot_convergence_grid.py \
  --raw-dir results/raws/armijo/mnist_qr_history \
  --output-dir results/figures/armijo/convergence_methods \
  --retractions qr \
  --ranks 32,64,128 \
  --lmbdas 1e-2,1e-4,1e-6 \
  --metrics rmse
```

## Summaries and paired tests

Grid summaries can be regenerated from raw records:

```bash
PYTHONPATH=src python experiments/summaries/armijo/summarize_armijo_convergence_methods.py \
  --raw-root results/raws/armijo \
  --run-prefix mnist \
  --retraction all
```

The paired-test command pools matched configurations across explicitly supplied
dataset prefixes. Use `--help` to inspect the expected group sizes and output
options:

```bash
PYTHONPATH=src python experiments/summaries/armijo/paired_t_tests.py --help
```

## Tests

```bash
pytest tests/unit -q
pytest -q
```

To reproduce the reference tables and figures, run the experiments rather than
treating the included processed CSVs and PNGs as substitutes for execution.
Results can vary slightly with hardware, numerical libraries, and dependency
versions.
