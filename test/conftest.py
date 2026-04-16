import numpy as np
import pytest

from heavy_ball import HeavyBall
from quadratic import Quadratic
from gradient_descent import GradientDescent


@pytest.fixture
def small_quadratic() -> Quadratic:
    A = np.array([[3.0, 1.0], [1.0, 2.0]])  # positive definite
    b = np.array([1.0, 1.0])
    return Quadratic(A, b)


@pytest.fixture
def gradient_descent_optimizer() -> GradientDescent:
    return GradientDescent(lr=0.1)


@pytest.fixture
def heavy_ball_optimizer() -> HeavyBall:
    return HeavyBall(lr=0.1, momentum=0.9)


@pytest.fixture
def x0() -> np.ndarray:
    return np.array([2.0, 2.0])
