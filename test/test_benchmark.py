import numpy as np
import pytest

from benchmark import run_benchmark
from gradient_descent import GradientDescent
from heavy_ball import HeavyBall


N_ITERS = 20


def test_benchmark_result_fields(small_quadratic, gradient_descent_optimizer, x0):
    result = run_benchmark(gradient_descent_optimizer, small_quadratic, x0, n_iters=N_ITERS)
    assert len(result.loss_history) == N_ITERS + 1
    assert len(result.grad_norm_history) == N_ITERS + 1
    assert result.optimizer_name == repr(gradient_descent_optimizer)
    assert result.problem_name == repr(small_quadratic)


def test_gd_loss_non_increasing_on_quadratic(small_quadratic, x0):
    result = run_benchmark(GradientDescent(lr=0.01), small_quadratic, x0, n_iters=N_ITERS)
    assert np.all(np.diff(result.loss_history) <= 1e-10)


def test_store_iterates_true(small_quadratic, gradient_descent_optimizer, x0):
    result = run_benchmark(gradient_descent_optimizer, small_quadratic, x0, n_iters=N_ITERS, store_iterates=True)
    assert len(result.x_history) == N_ITERS + 1


def test_store_iterates_false(small_quadratic, gradient_descent_optimizer, x0):
    result = run_benchmark(gradient_descent_optimizer, small_quadratic, x0, n_iters=N_ITERS, store_iterates=False)
    assert len(result.x_history) == 0


def test_gd_converges_on_quadratic(small_quadratic, x0):
    result = run_benchmark(GradientDescent(lr=0.01), small_quadratic, x0, n_iters=500)
    suboptimality = result.loss_history[-1] - small_quadratic.optimal_value()
    assert suboptimality < 1e-6


def test_benchmark_is_reproducible(small_quadratic, heavy_ball_optimizer, x0):
    result1 = run_benchmark(heavy_ball_optimizer, small_quadratic, x0, n_iters=N_ITERS)
    result2 = run_benchmark(heavy_ball_optimizer, small_quadratic, x0, n_iters=N_ITERS)
    np.testing.assert_array_equal(result1.loss_history, result2.loss_history)
