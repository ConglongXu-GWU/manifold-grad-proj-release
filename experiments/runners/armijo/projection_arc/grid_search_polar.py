from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.shared.armijo_grid import main_for_routine_and_retraction


ROUTINE = "armijo_projection_arc"
RETRACTION = "polar"


def main() -> None:
    main_for_routine_and_retraction(
        ROUTINE,
        RETRACTION,
        description="Checkpointed grid search for projection-arc Armijo with polar retraction.",
        default_run_name="armijo_projection_arc_polar_grid",
    )


if __name__ == "__main__":
    main()
