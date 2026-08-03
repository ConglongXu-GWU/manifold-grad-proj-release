from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runners.shared.armijo_grid import build_arg_parser, run_from_args


RETRACTION = "polar"


def main() -> None:
    parser = build_arg_parser(
        "Checkpointed grid search for Armijo routines with polar retraction.",
        include_retraction=False,
    )
    args = parser.parse_args()
    run_from_args(args, retraction=RETRACTION)


if __name__ == "__main__":
    main()
