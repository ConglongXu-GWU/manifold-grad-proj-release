"""Export full-training Stage 1 table data from exactly six grid shards.

Assumes the reported exhaustive grid: 27 candidates per routine/configuration,
200 iterations, ranks 32/64/128, and regularization 1e-2/1e-4/1e-6.
Raw records are read-only. Histories sharing the prefix are never selected.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.summaries.armijo.paired_t_tests import expected_grid_shard_dirs
from experiments.summaries.armijo.summarize_armijo_convergence_methods import (
    load_json_records,
    manuscript_group_summaries,
)

DEFAULT_PREFIX = 'cifar10gray50000_fulltrain_mask050_20260812'
FIELDS = ('retraction', 'rank', 'lmbda', 'routine', 'best_rmse', 'mean_rmse',
          'failures', 'successes', 'beta', 'sigma', 'initial_step',
          'final_wlra_loss', 'best_source')


def table_rows(raw_root: Path, run_prefix: str) -> list[dict]:
    records = []
    for shard in expected_grid_shard_dirs(raw_root=raw_root, run_prefix=run_prefix):
        rows = load_json_records(shard)
        if len(rows) != 243 or any(row.get('iteration_numbers') != 200 for row in rows):
            raise ValueError(f'Expected 243 Stage 1 records of 200 iterations: {shard.name}')
        records.extend(rows)
    summaries = manuscript_group_summaries(records)
    counts = {}
    for row in records:
        key = (row['retraction'], row['rank'], row['lmbda'], row['config']['routine'])
        counts[key] = counts.get(key, 0) + 1
    if len(summaries) != 54 or any(count != 27 for count in counts.values()):
        raise ValueError('Expected 54 routine/configuration groups of 27 candidates')
    output = []
    for key, summary in sorted(summaries.items()):
        best = summary.best
        row = dict(retraction=key[0], rank=key[1], lmbda=key[2], routine=key[3],
                   best_rmse=None if best is None else best['final_rmse'],
                   mean_rmse=summary.mean_rmse, failures=summary.failures,
                   successes=counts[key] - summary.failures)
        row.update({field: None if best is None else best['config'][field]
                    for field in ('beta', 'sigma', 'initial_step')})
        row['final_wlra_loss'] = None if best is None else best['final_wlra_loss']
        row['best_source'] = '' if best is None else str(Path('results/raws/armijo') / Path(best['_source_file']).resolve().relative_to(raw_root.resolve()))
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'results/raws/armijo')
    parser.add_argument('--run-prefix', default=DEFAULT_PREFIX)
    parser.add_argument('--output-csv', type=Path, default=ROOT / 'results/processed/armijo/full_cifar10_stage1_table_data.csv')
    args = parser.parse_args()
    rows = table_rows(args.raw_root, args.run_prefix)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
