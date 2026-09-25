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

This is the standalone code release, not a mirror of the research repository.
Manuscripts, AAAI archives, private evidence, reviewer notes, agent files,
dataset binaries and raw records are not included. Obtain the public datasets
and regenerate local raw records using the commands below.

## Repository structure

```text
manifold-grad-proj-release/
|-- README.md                  Installation and reproduction instructions
|-- LICENSE                    Software license
|-- pyproject.toml             Package and test configuration
|-- requirements.txt           Runtime dependencies
|-- .github/workflows/         Automated checks
|-- src/manifold_opt/          Geometry, objectives, line searches, optimizers
|-- data_preparation/          Dataset download and preprocessing tools
|-- experiments/
|   |-- runners/              Grid searches and selected long runs
|   |-- summaries/            Result aggregation and paired tests
|   `-- plotting/             Convergence figures
|-- results/
|   |-- processed/            Approved small reference summaries
|   `-- figures/              Reference figures
`-- tests/
    |-- unit/                 Component and regression tests
    `-- integration/          End-to-end workflow tests
```

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

## Full-training Stage 1 reproduction

Prepare the full MNIST training split (60,000 columns, 784 rows) and grayscale
CIFAR-10 training split (50,000 columns, 1024 rows), both at masking rate 0.5
and preprocessing seed 42. The CIFAR reference used the `uoft-cs/cifar10`
Hugging Face parquet training split; obtain its
`plain_text/train-00000-of-00001.parquet` file and place it at the path below.
The loader preserves image order and converts each image to grayscale.

```bash
python -m data_preparation.prepare_mnist \
  --full-train --mask-rates 0.50 --seeds 42 --output-dir data
python -m data_preparation.prepare_cifar10 \
  --source hf-parquet \
  --hf-train-parquet data/cifar10-hf/plain_text/train-00000-of-00001.parquet \
  --full-train --mask-rates 0.50 --seeds 42 --output-dir data
```

Run six distinct shards per dataset. The loop below runs them sequentially;
independent shards may run concurrently if memory and CPU resources permit.
Keep the shard names distinct, and do not use `--force` when resuming.
These are expensive full-training experiments, not quick installation tests.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
for dataset in mnist cifar10; do
  if [ "$dataset" = mnist ]; then
    artifact=data/mnist_train-full_mask0.50_seed42.pt
    prefix=full_train_20260618
  else
    artifact=data/cifar10-gray_train-full_mask0.50_seed42.pt
    prefix=cifar10gray50000_fulltrain_mask050_20260812
  fi
  for retraction in qr polar; do
    for label in regularized projection_arc feasible_direction; do
      routine=armijo_$label
      if [ "$label" = regularized ]; then routine=armijo_wlra_reg; fi
      PYTHONPATH=src python experiments/runners/armijo/grid_search_${retraction}.py \
        --dataset-path "$artifact" --run-name "${prefix}_${retraction}_${label}" \
        --routines "$routine" --betas 0.5,0.75,0.9 --sigmas 0.25,0.5,0.75 \
        --initial-steps 0.3,0.5,1.0 --max-backtracks 200 \
        --ranks 32,64,128 --lmbdas 1e-2,1e-4,1e-6 --iterations 200 \
        --r auto --rmse-mode final --device cpu --dtype float32
    done
  done
done
```

Each shard has 243 records; each dataset has 1,458. The full-CIFAR reference
contains 1,276 successful runs and 182 failures, with 27 Armijo grid points
per routine/rank/regularization/retraction configuration. The approved summary
is `results/processed/armijo/full_cifar10_stage1_table_data.csv`.
Regenerate that summary from the six local raw shards with:

```bash
PYTHONPATH=src python experiments/summaries/armijo/summarize_full_train_stage1.py
```

Means exclude failed runs; failures are reported separately. Best candidates
minimize final RMSE, with final unregularized WLRA loss breaking ties.
The backtracking cap is 200. When reported, mean Armijo evaluations count
actual inequality checks per iteration, averaged over successful grid runs;
they are neither total run counts nor wall-clock costs.

The six sampled Stage 1 dataset prefixes used in the reference analysis are:

| Dataset | Mask 0.5 | Mask 0.75 | Mask 0.25 |
| --- | --- | --- | --- |
| MNIST, 6,000 images | `digits6000_20260707` | `digits6000_mask075_20260712` | `digits6000_mask025_20260718` |
| CIFAR-10, 5,000 images | `cifar10gray5000_20260708` | `cifar10gray5000_mask075_20260712` | `cifar10gray5000_mask025_20260718` |

For each sampled artifact, use the same exhaustive grid options and six-shard
naming convention above with the corresponding prefix. The shorter example
in the preceding section is only a smoke-run configuration.

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

For the expanded Stage 1 analysis, regenerate all eight groups and supply every
prefix explicitly. This produces 144 complete pairs (143 degrees of freedom)
for each of four one-sided tests, using the unadjusted 0.05 threshold. The
reference p-values are approximately 0.0061, 0.0071, 0.0108 and 0.0235, in CSV
row order. The legacy default is seven groups/126 pairs; do not use that
default to reproduce this expanded analysis.

```bash
PYTHONPATH=src python experiments/summaries/armijo/paired_t_tests.py \
  --analysis grid-search --expected-pairs 144 \
  --dataset-prefix 'Sampled MNIST (masking rate 0.5)' digits6000_20260707 \
  --dataset-prefix 'Sampled CIFAR-10 (masking rate 0.5)' cifar10gray5000_20260708 \
  --dataset-prefix 'Sampled MNIST (masking rate 0.75)' digits6000_mask075_20260712 \
  --dataset-prefix 'Sampled CIFAR-10 (masking rate 0.75)' cifar10gray5000_mask075_20260712 \
  --dataset-prefix 'Sampled MNIST (masking rate 0.25)' digits6000_mask025_20260718 \
  --dataset-prefix 'Sampled CIFAR-10 (masking rate 0.25)' cifar10gray5000_mask025_20260718 \
  --dataset-prefix 'Full-training MNIST (masking rate 0.5)' full_train_20260618 \
  --dataset-prefix 'Full-training CIFAR-10 (masking rate 0.5)' cifar10gray50000_fulltrain_mask050_20260812 \
  --output-csv results/processed/armijo/all_grid_search_paired_t_tests.csv \
  --output-tex results/processed/armijo/all_grid_search_paired_t_tests.tex
```

The loader selects exactly six Stage 1 shards per dataset, excluding any
similarly named 1,000-iteration histories. The **separate long-run analysis
remains 108 pairs** from six sampled dataset/masking groups, not the two full
training splits. Its reference CSV and figures are unchanged. After reproducing
those histories with the six default long-run prefixes shown by the script:

```bash
PYTHONPATH=src python experiments/summaries/armijo/paired_t_tests.py \
  --analysis long-run --expected-pairs 108 \
  --output-csv results/processed/armijo/all_long_run_final_rmse_paired_t_tests.csv \
  --output-tex results/processed/armijo/all_long_run_final_rmse_paired_t_tests.tex
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
