from pathlib import Path

import pytest

from experiments.summaries.armijo import summarize_full_train_stage1 as summary


def test_rejects_incomplete_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'expected_grid_shard_dirs', lambda **kwargs: (tmp_path,))
    monkeypatch.setattr(summary, 'load_json_records', lambda path: [])
    with pytest.raises(ValueError, match='243 Stage 1'):
        summary.table_rows(tmp_path, 'grid')


def test_uses_exact_shards_and_successful_grid_aggregation(tmp_path, monkeypatch):
    shards = [tmp_path / f'{retraction}_{routine}' for retraction in ('qr', 'polar')
              for routine in ('armijo_wlra_reg', 'armijo_feasible_direction', 'armijo_projection_arc')]
    monkeypatch.setattr(summary, 'expected_grid_shard_dirs', lambda **kwargs: shards)
    loaded = []
    def load(shard):
        loaded.append(shard)
        retraction, routine = shard.name.split('_', 1)
        return [dict(retraction=retraction, rank=rank, lmbda=lmbda,
                     config=dict(routine=routine, beta=0.5, sigma=0.25, initial_step=0.3),
                     iteration_numbers=200, status='failed' if i == 26 else 'success',
                     final_rmse=0.1, final_wlra_loss=27-i,
                     _source_file=str(shard / f'{rank}_{lmbda}_{i}.json'))
                for rank in (32, 64, 128) for lmbda in (1e-2, 1e-4, 1e-6) for i in range(27)]
    monkeypatch.setattr(summary, 'load_json_records', load)
    rows = summary.table_rows(tmp_path, 'grid')
    assert loaded == shards and len(rows) == 54
    assert all(r['failures'] == 1 and r['successes'] == 26 for r in rows)
    assert all(r['mean_rmse'] == pytest.approx(0.1) for r in rows)
    assert all(r['final_wlra_loss'] == 2 for r in rows)
    assert all(not Path(r['best_source']).is_absolute() for r in rows)
