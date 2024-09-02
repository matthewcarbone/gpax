"""
acquisition.py
==============

Base acquisition functions and objects

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro.distributions as dist
from monty.json import MSONable


def expected_improvement(mean, variance, y_max):
    """Expected Improvement acquisition function.

    Parameters
    ----------
    mean, variance : array_like
        The predictive mean (variance) at some point or set of points.
    y_max : float
        The current largest value observed by the model.

    Returns
    -------
    jnp.ndarray
    """

    sigma = jnp.sqrt(variance)
    u = (mean - y_max) / sigma
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    ucdf = normal.cdf(u)
    updf = jnp.exp(normal.log_prob(u))
    acq = sigma * (updf + u * ucdf)
    return acq


def upper_confidence_bound(mean, variance, beta=0.25):
    """Upper Confidence Bound acquisition function.

    Parameters
    ----------
    mean, variance : array_like
        The predictive mean (variance) at some point or set of points.
    beta : float
        Coefficient for balancing the exploration-exploitation tradeoff. If
        beta is set to float("inf") or similar, upper_confidence_bound
        reduces to pure uncertainty-based exploration.

    Returns
    -------
    jnp.ndarray
    """

    if bool(jnp.isinf(beta)):
        return jnp.sqrt(variance)

    return mean + jnp.sqrt(beta * variance)


def probability_of_improvement(mean, variance, best_f=None, xi=0.01):
    """Probability of Improvement acquisition function.

    Parameters
    ----------
    mean, variance : array_like
        The predictive mean (variance) at some point or set of points.
    best_f : float, optional
        Best function value observed so far. Derived from the predictive mean
        when not provided.
    xi : float
        Balances the exploration-exploitation tradeoff.

    Returns
    -------
    jnp.ndarray
    """

    if best_f is None:
        best_f = mean.max()
    sigma = jnp.sqrt(variance)
    u = (mean - best_f - xi) / sigma
    normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
    return normal.cdf(u)
