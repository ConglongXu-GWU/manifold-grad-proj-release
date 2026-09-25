"""Default derived outputs must not recreate retired manuscript directories."""
from experiments.summaries.armijo import paired_t_tests, summarize_armijo_convergence_methods


def test_summary_defaults_are_outside_manuscript():
    root = paired_t_tests.REPO_ROOT / 'results' / 'processed' / 'armijo'
    for path in (
        paired_t_tests.DEFAULT_OUTPUT_TEX,
        paired_t_tests.DEFAULT_LONG_RUN_OUTPUT_TEX,
        summarize_armijo_convergence_methods.DEFAULT_MANUSCRIPT_TABLE,
    ):
        assert path.parent == root
