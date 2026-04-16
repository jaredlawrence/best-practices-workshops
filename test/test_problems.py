import numpy as np
import pytest

from quadratic import Quadratic
from rosenbrock import Rosenbrock
from utils import finite_difference_gradient


@pytest.fixture
def rosenbrock() -> Rosenbrock:
    return Rosenbrock(dim=2)


# --- Gradient accuracy ---

def test_quadratic_gradient_matches_finite_difference(small_quadratic, x0):
    analytic = small_quadratic.gradient(x0)
    numerical = finite_difference_gradient(small_quadratic, x0)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4)


def test_rosenbrock_gradient_matches_finite_difference(rosenbrock, x0):
    analytic = rosenbrock.gradient(x0)
    numerical = finite_difference_gradient(rosenbrock, x0)
    np.testing.assert_allclose(analytic, numerical, atol=1e-4)


# --- Quadratic optimum ---

def test_quadratic_evaluate_at_optimum_equals_optimal_value(small_quadratic):
    assert small_quadratic.evaluate(small_quadratic.optimum()) == small_quadratic.optimal_value()


def test_quadratic_gradient_norm_at_optimum_is_small(small_quadratic):
    grad_norm = np.linalg.norm(small_quadratic.gradient(small_quadratic.optimum()))
    assert grad_norm < 1e-6


# --- Rosenbrock optimum ---

def test_rosenbrock_optimum_is_ones(rosenbrock):
    np.testing.assert_array_equal(rosenbrock.optimum(), np.ones(2))


def test_rosenbrock_gradient_norm_at_optimum_is_small(rosenbrock):
    grad_norm = np.linalg.norm(rosenbrock.gradient(rosenbrock.optimum()))
    assert grad_norm < 1e-5
