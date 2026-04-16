import numpy as np

from optimizer_base import Optimizer


class GradientDescent(Optimizer):
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return x - self.lr * grad

    def reset(self) -> None:
        pass
