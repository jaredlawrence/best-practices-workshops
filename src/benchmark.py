from dataclasses import dataclass

import numpy as np

from optimizer_base import Optimizer
from utils import Timer


@dataclass
class BenchmarkResult:
    optimizer_name: str
    problem_name: str
    loss_history: np.ndarray
    grad_norm_history: np.ndarray
    x_history: list[np.ndarray]
    runtime_seconds: float
    n_iters: int


def run_benchmark(
    optimizer: Optimizer,
    problem,
    x0: np.ndarray,
    n_iters: int,
    store_iterates: bool = False,
) -> BenchmarkResult:
    """Run an optimizer on a problem for n_iters steps from x0.

    Parameters
    ----------
    optimizer:
        Any object implementing ``step`` and ``reset``.
    problem:
        Any object implementing ``evaluate`` and ``gradient``.
    x0:
        Starting point.
    n_iters:
        Number of gradient steps to take.
    store_iterates:
        If True, record the full iterate sequence in ``x_history``.
    """
    x = x0.copy()
    optimizer.reset()

    losses, grad_norms, iterates = [], [], []

    with Timer() as timer:
        for i in range(n_iters + 1):
            grad = problem.gradient(x)
            losses.append(problem.evaluate(x))
            grad_norms.append(float(np.linalg.norm(grad)))
            if store_iterates:
                iterates.append(x.copy())
            if i < n_iters:
                x = optimizer.step(x, grad)

    return BenchmarkResult(
        optimizer_name=repr(optimizer),
        problem_name=repr(problem),
        loss_history=np.array(losses),
        grad_norm_history=np.array(grad_norms),
        x_history=iterates,
        runtime_seconds=timer.elapsed,
        n_iters=n_iters,
    )
