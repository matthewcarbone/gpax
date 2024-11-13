"""
Utility functions used in GPax.
"""

# Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
# Modified by Matthew R. Carbone (email: x94carbone@gmail.com)

from collections.abc import Iterable
from itertools import product
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


def enable_x64() -> None:
    """Use double (x64) precision for jax arrays

    :::{important}
    Due to how sensitive the kernel operations can be on the floating point
    precision, x64 is enabled automatically whenever {py:class}`gpax.gp` is
    imported. If you want to disable this behavior, you can do so manually via

    ```python
    import jax

    jax.config.update("jax_enable_x64", False)
    ```
    :::
    """

    jax.config.update("jax_enable_x64", True)


def split_array(x: Iterable, s: int = 100) -> list[Any]:
    """Splits an iterable `x` along its 0th axis into parts [roughly] equal to
    the provided batch size, `s`

    :param x:
        The iterable to split along its 0th axis.
    :param s:
        The batch size. This is the approximate size each of the new list
        chunks will be.

    :returns: A new list in which the provided one has been split.
    """

    return [x[i : i + s, ...] for i in range(0, len(x), s)]


def scale_by_domain(x: ArrayLike, domain: ArrayLike) -> ArrayLike:
    """Scales provided data `x` by the bounds provided in the `domain`

    :::{note}
    All dimensions must *perfectly* match.
    :::

    :param x:
        The input data of shape `(N, d)`, scaled between 0 and 1 along all
        dimensions.
    :param domain:
        The domain to scale to. Should be of shape `(2, d)`, where
        `domain[0, :]` is the minimum along each axis and `domain[1, :]`
        is the maximum.

    :returns: The scaled data of shape `(N, d)`.

    :raises ValueError: If any dimensions do not match, or if the input data
    is not scaled properly between 0 and 1 along all dimensions.
    """

    if x.ndim != 2:
        raise ValueError("Dimension of x must be == 2")

    if domain.shape[1] != x.shape[1]:
        raise ValueError("Domain and x shapes mismatched")

    if domain.shape[0] != 2:
        raise ValueError("Domain shape not equal to 2")

    if x.min() < 0.0:
        raise ValueError("x.min() < 0 (should be >= 0)")

    if x.max() > 1.0:
        raise ValueError("x.max() > 0 (should be <= 0)")

    return (domain[1, :] - domain[0, :]) * x + domain[0, :]


def get_coordinates(
    points_per_dimension: int | list, domain: np.typing.ArrayLike
) -> np.typing.ArrayLike:
    """Gets a grid of equally spaced points on each dimension and returns
    these results in coordinate representation

    :param points_per_dimension:
        The number of points per dimension. If int, assumed to be 1d.
    :param domain:
        A shape `(2, d)` array indicating the domain along each axis.

    :returns: The available points for sampling.
    """

    if isinstance(points_per_dimension, int):
        points_per_dimension = [points_per_dimension] * domain.shape[1]
    gen = product(*[jnp.linspace(0.0, 1.0, nn) for nn in points_per_dimension])
    return scale_by_domain(jnp.array([xx for xx in gen]), domain)


class Timer:
    """A simple timer context manager

    # Usage

    ```python
    import time

    with Timer() as timer:
        time.sleep(5)
    print(f"done in {timer.elapsed:.02f} seconds")
    ```
    """

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        tf = perf_counter()
        self.elapsed = tf - self.t0


# def time_function(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with Timer() as timer:
#             result = func(*args, **kwargs)
#         if state.verbose > 1:
#             print(f"{func.__qualname__} took {timer.elapsed:.02e} s")
#         return result
#
#     return wrapper
#

# def delay_warnings(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with warnings.catch_warnings(record=True) as captured_warnings:
#             warnings.simplefilter("always")
#             result = func(*args, **kwargs)
#         seen_warnings = set()
#         unique_warnings = []
#         for w in captured_warnings:
#             key = (str(w.message), w.category, w.filename, w.lineno, w.line)
#             if key not in seen_warnings:
#                 unique_warnings.append(w)
#         for w in unique_warnings:
#             warnings.showwarning(
#                 message=w.message,
#                 category=w.category,
#                 filename=w.filename,
#                 lineno=w.lineno,
#                 file=sys.stderr,
#                 line=w.lilne,
#             )
#         return result
#
#     return wrapper


# def dict_of_list_to_list_of_dict(d):
#     """Emulates a lot of the functionality of jax.vmap but in pure Python.
#     Converts a dictionary of arrays to a list of dictionaries in which each
#     key of the resultant dictionaries contains only a single value. For
#     example, converts
#
#     data = {
#         'a': [1, 2, 3],
#         'b': [4, 5, 6],
#         'c': [7, 8, 9],
#     }
#
#     to
#
#     [
#         {'a': 1, 'b': 4, 'c': 7},
#         {'a': 2, 'b': 5, 'c': 8},
#         {'a': 3, 'b': 6, 'c': 9}
#     ]
#     """
#
#     k = d.keys()
#     v = zip(*d.values())
#     for vv in v:
#         yield dict(zip(k, vv))
