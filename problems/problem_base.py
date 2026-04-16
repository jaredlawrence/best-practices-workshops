from abc import ABC, abstractmethod

import numpy as np


class Problem(ABC):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Return the scalar function value at x."""
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Return the gradient vector at x."""
        ...

    @abstractmethod
    def optimum(self) -> np.ndarray:
        """Return the known minimizer x*."""
        ...

    def optimal_value(self) -> float:
        return self.evaluate(self.optimum())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
