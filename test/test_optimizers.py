import numpy as np

from gradient_descent import GradientDescent
from heavy_ball import HeavyBall


def test_gd_zero_lr_returns_same_point(x0):
    np.testing.assert_array_equal(GradientDescent(lr=0.0).step(x0, np.array([1.0, 2.0])), x0)


def test_heavy_ball_zero_momentum_matches_gd(x0):
    lr, grad = 0.1, np.array([1.0, 2.0])
    np.testing.assert_allclose(
        HeavyBall(lr=lr, momentum=0.0).step(x0, grad),
        GradientDescent(lr=lr).step(x0, grad),
    )


def test_heavy_ball_reset_clears_velocity(heavy_ball_optimizer, x0):
    heavy_ball_optimizer.step(x0, np.array([1.0, 1.0]))
    assert heavy_ball_optimizer.velocity is not None
    heavy_ball_optimizer.reset()
    assert heavy_ball_optimizer.velocity is None

