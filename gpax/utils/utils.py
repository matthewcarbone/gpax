"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from contextlib import contextmanager
from functools import wraps
from itertools import product
from time import perf_counter

import jax
import jax.numpy as jnp


def enable_x64():
    """Use double (x64) precision for jax arrays"""
    jax.config.update("jax_enable_x64", True)


def split_array(x, s=100):
    """Splits an array along its 0th axis into parts [roughly] equal to the
    provided batch size, s."""

    return [x[i : i + s, ...] for i in range(0, len(x), s)]


def scale_by_domain(x, domain):
    """Scales provided data x by the bounds provided in the domain. Note that
    all dimensions must perfectly match.

    Parameters
    ----------
    x : array_like
        The input data of shape (N, d). This data should be scaled between 0
        and 1.
    domain : array_like
        The domain to scale to. Should be of shape (2, d), where domain[0, :]
        is the minimum along each axis and domain[1, :] is the maximum.

    Returns
    -------
    array_like
        The scaled data of shape (N, d).
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


def get_coordinates(points_per_dimension, domain):
    """Gets a grid of equally spaced points on each dimension.
    Returns these results in coordinate representation.

    Parameters
    ----------
    points_per_dimension : int or list
        The number of points per dimension. If int, assumed to be 1d.
    domain : array_like
        A 2 x d array indicating the domain along each axis.

    Returns
    -------
    jnp.ndarray
        The available points for sampling.
    """

    if isinstance(points_per_dimension, int):
        points_per_dimension = [points_per_dimension] * domain.shape[1]
    gen = product(*[jnp.linspace(0.0, 1.0, nn) for nn in points_per_dimension])
    return scale_by_domain(jnp.array([xx for xx in gen]), domain)


@contextmanager
def Timer():
    start = perf_counter()
    yield lambda: perf_counter() - start


def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer() as dt:
            result = func(*args, **kwargs)
        print(f"{func.__qualname__} took {dt():.02e} s")
        return result

    return wrapper


def dict_of_list_to_list_of_dict(d):
    """Emulates a lot of the functionality of jax.vmap but in pure Python.
    Converts a dictionary of arrays to a list of dictionaries in which each
    key of the resultant dictionaries contains only a single value. For
    example, converts

    data = {
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9],
    }

    to

    [
        {'a': 1, 'b': 4, 'c': 7},
        {'a': 2, 'b': 5, 'c': 8},
        {'a': 3, 'b': 6, 'c': 9}
    ]
    """

    k = d.keys()
    v = zip(*d.values())
    for vv in v:
        yield dict(zip(k, vv))
