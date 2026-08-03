from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.shared.armijo_grid import main_for_routine


ROUTINE = "armijo_wlra_reg"


def main() -> None:
    main_for_routine(
        ROUTINE,
        description="Checkpointed grid search for regularized WLRA Armijo.",
        default_run_name="armijo_wlra_reg_grid",
    )


if __name__ == "__main__":
    main()
