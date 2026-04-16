"""Read saved CSVs and produce comparison figures."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "src"), str(ROOT / "problems")])

from plotting import plot_convergence, plot_grad_norm


def load_results(csv_dir: Path, problem: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Load all CSVs for a given problem; return labelled dict and problem stem."""
    results = {}
    problem_stem = ""
    for path in sorted(csv_dir.glob(f"*_{problem}_*.csv")):
        df = pd.read_csv(path)
        label = df["optimizer"].iloc[0]
        problem_stem = df["problem"].iloc[0]
        results[label] = df
    if not results:
        raise FileNotFoundError(
            f"No CSVs found for problem '{problem}' in {csv_dir}. "
            "Run run_benchmark.py first."
        )
    return results, problem_stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved benchmark results.")
    parser.add_argument("--problem",    type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results, problem_stem = load_results(csv_dir, args.problem)
    print(f"Loaded {len(results)} result(s) for '{problem_stem}':")
    for label in results:
        print(f"  {label}")

    stem = f"comparison_{problem_stem}"
    plot_convergence(results, title=problem_stem, save_path=str(fig_dir / f"{stem}_convergence.png"))
    plot_grad_norm(results,   title=problem_stem, save_path=str(fig_dir / f"{stem}_grad_norm.png"))
    print(f"\nFigures: {fig_dir}/")


if __name__ == "__main__":
    main()