from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    @abstractmethod
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Take one gradient step; return the updated iterate."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Restore internal state to its initial value."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr})"
