import sys
from warnings import warn

import jax

_this = sys.modules[__name__]


# TODO: multi-gpu support???
def set_device(device):
    try:
        _this.device = jax.devices(device)[0]
    except RuntimeError as err:
        warn(
            f"Device {device} not detected on your machine. Jax sent the "
            f"following error message: '{err}'. Device set to cpu as fallback."
        )
        _this.device = jax.devices("cpu")[0]


set_device("cpu")
