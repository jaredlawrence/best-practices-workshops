import numpy as np

from problem_base import Problem


class Quadratic(Problem):
    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        super().__init__(dim=A.shape[0])
        self.A = A
        self.b = b

    def evaluate(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.A @ x - self.b @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def optimum(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)
