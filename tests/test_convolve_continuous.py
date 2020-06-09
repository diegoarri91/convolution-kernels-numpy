import numpy as np
import pytest

from kernel.fun import KernelFun


@pytest.fixture()
def exponential_kernel_white_noise():
    dt = 0.2
    td = 10

    support = np.array([0, 10 * td])
    ker = KernelFun.single_exponential(td, support=support)
    # t_int = np.arange(0, 10 * td, dt)
    # ker_vals = ker.interpolate(t_int)

    t = np.arange(0, 500, dt)

    x = np.random.randn(len(t), 1)
    y = ker.convolve_continuous(t, x)

    return ker, t, x, y


def test_convolve_continuous(exponential_kernel_white_noise):
    ker, t, x, y_true = exponential_kernel_white_noise
    dt = t[1]
    ker_vals = ker.interpolate(np.arange(ker.support[0], ker.support[1], dt))
    y = np.array(
        [np.sum(ker_vals[:min(u + 1, len(ker_vals))][::-1] * \
                x[max(0, u + 1 - len(ker_vals)):u + 1, 0]) \
         for u in range(len(t))])
    y = y * dt
    assert np.all((y_true[:, 0] - y) < 5e-5)
