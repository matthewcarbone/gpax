import sys
from warnings import warn

import jax
import numpy as np

_this = sys.modules[__name__]


# TODO: multi-gpu support?
def set_device(device):
    try:
        _this.device = jax.devices(device)[0]
    except RuntimeError as err:
        warn(
            f"Device {device} not detected on your machine. Jax sent the "
            f"following error message: '{err}'. Device set to cpu as fallback."
        )
        _this.device = jax.devices("cpu")[0]


def set_rng_key(key):
    assert isinstance(key, int)
    assert key >= 0
    assert key <= 2**32 - 1
    _this.rng_key = key


def get_rng_key():
    """A utility for retrieving the RNG state of gpax. This comes in the form
    of two keys: an integer used for seeding e.g. numpy, scipy, etc. and a
    jax key, which comes in its own form.

    Every time the function is called, the RNG is iterated. The current key
    state is used as a seed for the next keys.

    Increment the rng_key in a pseudo-random way (we don't just increment
    the rng_key by one, instead, we use the deterministic pseudo-random
    jax.random.split method to produce a "random-ish" next state. This
    ensures that setting key = 1 produces a truly random process from
    key = 2. In other words, if we just incremented the state, we'd have
    overlap, e.g.
    key  = 1 -> 2 -> 3
    key' = 2 -> 3 -> 4
    (2 and 3 overlap)
    whereas using jra.random.split produces
    key  = 1 -> 1948878966 -> 2215249346
    key' = 2 -> 637334850  -> 2584188353
    """

    key = _this.rng_key

    # Get a new numpy key
    np.random.seed(key)
    new_key = np.random.randint(low=0, high=2**32 - 1)
    new_jax_key = jax.random.key(new_key)

    _this.rng_key = new_key

    return new_key, new_jax_key


set_device("cpu")
set_rng_key(0)
