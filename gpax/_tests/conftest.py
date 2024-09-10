import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def dummy_data():
    np.random.seed(123)
    N = 100
    x = (jnp.array(np.random.random(size=(N, 2))) * 10 / 2) ** 2 + 2
    y = 10 * x**2
    return jnp.array(x), jnp.array(y)


@pytest.fixture
def dummy_sinusoidal_data():
    np.random.seed(0)

    NUM_INIT_POINTS = 25  # number of observation points
    NOISE_LEVEL = 0.1  # noise level

    f = lambda x: np.sin(10 * x)

    x = np.random.uniform(-1.0, 1.0, NUM_INIT_POINTS)
    y = f(x) + np.random.normal(0.0, NOISE_LEVEL, NUM_INIT_POINTS)
    y = y.squeeze()
    x = x.reshape(-1, 1)
    x_grid = np.linspace(-1.1, 1.1, 300).reshape(-1, 1)

    return x, y, x_grid
