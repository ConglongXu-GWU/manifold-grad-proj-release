from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from experiments.plotting.shared.convergence_common import create_polar_three_method_convergence_plots, parse_common_args
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from experiments.plotting.shared.convergence_common import create_polar_three_method_convergence_plots, parse_common_args


RETRACTION = "polar"


def main() -> None:
    args = parse_common_args("Plot polar-retraction WLRA convergence for constrained and regularized Armijo methods.")
    create_polar_three_method_convergence_plots(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        k=args.rank,
        r=args.r,
        lmbda=args.lmbda,
        lr=args.lr,
        iteration_numbers=args.iterations,
        beta=args.beta,
        sigma=args.sigma,
        max_backtracks=args.max_backtracks,
        armijo_tol=args.armijo_tol,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
