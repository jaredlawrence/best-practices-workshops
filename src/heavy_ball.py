import numpy as np

from optimizer_base import Optimizer


class HeavyBall(Optimizer):
    def __init__(self, lr: float, momentum: float) -> None:
        super().__init__(lr)
        self.momentum = momentum
        self.velocity: np.ndarray | None = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return x + self.velocity

    def reset(self) -> None:
        self.velocity = None

    def __repr__(self) -> str:
        return f"HeavyBall(lr={self.lr}, momentum={self.momentum})"
