import time

import numpy as np

def finite_difference_gradient(problem, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Approximate the gradient of problem at x using central differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        grad[i] = (problem.evaluate(x + e) - problem.evaluate(x - e)) / (2 * eps)
    return grad


class Timer:
    """Context manager that records elapsed wall time in seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
