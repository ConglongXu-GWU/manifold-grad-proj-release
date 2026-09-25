from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.shared.constant_step_grid import main_for_routine


ROUTINE = "optimize_reg"


def main() -> None:
    main_for_routine(
        ROUTINE,
        description="Timestamped raw JSON grid search for regularized WLRA constant stepsizes.",
    )


if __name__ == "__main__":
    main()
