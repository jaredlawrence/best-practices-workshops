import numpy as np

from problem_base import Problem


class Rosenbrock(Problem):
    def __init__(self, dim: int = 2) -> None:
        super().__init__(dim=dim)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(self.dim - 1):
            grad[i]     += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1 - x[i])
            grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return grad

    def optimum(self) -> np.ndarray:
        return np.ones(self.dim)
