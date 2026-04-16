import matplotlib.pyplot as plt
import pandas as pd


def _save_or_show(save_path: str | None) -> None:
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_convergence(
    results: dict[str, pd.DataFrame],
    title: str,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots()
    for label, df in results.items():
        ax.semilogy(df["iteration"], df["suboptimality"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x) - f(x*)")
    ax.set_title(f"Suboptimality — {title}")
    ax.legend()
    _save_or_show(save_path)


def plot_grad_norm(
    results: dict[str, pd.DataFrame],
    title: str,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots()
    for label, df in results.items():
        ax.semilogy(df["iteration"], df["grad_norm"], label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    ax.set_title(f"Gradient norm — {title}")
    ax.legend()
    _save_or_show(save_path)
