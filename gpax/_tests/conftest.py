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
