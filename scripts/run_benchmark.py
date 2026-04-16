"""Run a single optimizer on a single problem and save results."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "src"), str(ROOT / "problems")])

from benchmark import run_benchmark
from gradient_descent import GradientDescent
from heavy_ball import HeavyBall
from quadratic import Quadratic
from rosenbrock import Rosenbrock


OPTIMIZERS = {
    "gradient_descent": lambda args: GradientDescent(lr=args.lr),
    "heavy_ball":       lambda args: HeavyBall(lr=args.lr, momentum=args.momentum),
}

PROBLEMS = {
    "quadratic":  lambda args: Quadratic(np.eye(args.dim) * args.A_scale,
                                         np.ones(args.dim) * args.b_scale),
    "rosenbrock": lambda args: Rosenbrock(dim=args.dim),
}

OPTIMIZER_STEMS = {
    "gradient_descent": lambda args: f"gradient_descent_lr{args.lr}",
    "heavy_ball":       lambda args: f"heavy_ball_lr{args.lr}_m{args.momentum}",
}

PROBLEM_STEMS = {
    "quadratic":  lambda args: f"quadratic_dim{args.dim}_A{args.A_scale}_b{args.b_scale}",
    "rosenbrock": lambda args: f"rosenbrock_dim{args.dim}",
}


def make_stems(args: argparse.Namespace) -> tuple[str, str]:
    """Return (optimizer_stem, problem_stem) for use in filenames and plot titles."""
    return OPTIMIZER_STEMS[args.optimizer](args), PROBLEM_STEMS[args.problem](args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single optimizer benchmark.")
    parser.add_argument("--optimizer",  choices=list(OPTIMIZERS), required=True)
    parser.add_argument("--problem",    choices=list(PROBLEMS),   required=True)
    parser.add_argument("--lr",         type=float, required=True)
    parser.add_argument("--momentum",   type=float, default=0.9)
    parser.add_argument("--n-iters",    type=int,   required=True)
    parser.add_argument("--dim",        type=int,   default=2)
    parser.add_argument("--A-scale",    type=float, default=2.0)
    parser.add_argument("--b-scale",    type=float, default=1.0)
    parser.add_argument("--output-dir", type=str,   default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    problem   = PROBLEMS[args.problem](args)
    optimizer = OPTIMIZERS[args.optimizer](args)
    x0        = np.zeros(args.dim)

    result = run_benchmark(optimizer, problem, x0, n_iters=args.n_iters)

    optimizer_stem, problem_stem = make_stems(args)
    stem = f"{optimizer_stem}_{problem_stem}_{args.n_iters}iters"

    df = pd.DataFrame({
        "iteration":     range(args.n_iters + 1),
        "loss":          result.loss_history,
        "suboptimality": result.loss_history - problem.optimal_value(),
        "grad_norm":     result.grad_norm_history,
        "optimizer":     optimizer_stem,
        "problem":       problem_stem,
    })
    csv_path = csv_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()