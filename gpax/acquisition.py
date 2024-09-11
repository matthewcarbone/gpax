"""
acquisition.py
==============

Base acquisition functions and objects

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable
from warnings import warn

import jax.numpy as jnp
import jax.random as jra
import jaxopt
import numpy as np
import numpyro.distributions as dist
from attrs import define, field
from attrs.validators import ge, gt, instance_of
from monty.json import MSONable
from scipy.stats import qmc
from tqdm import tqdm

from gpax.utils import split_array

DATA_TYPES = [jnp.ndarray, np.ndarray]


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


# def optimize_call_processing(fn):
#     @wraps(fn)
#     def inner(rng_key, x, **kwargs):
#         shape = x.shape
#         if x.ndim > 2:
#             x = x.reshape(-1, x.shape[-1])
#         response = fn(rng_key, x, **kwargs)
#         if hasattr(response, "__len__"):
#             response = [y.reshape(shape[:-1]) for y in response]
#         else:
#             response = response.reshape(shape[:-1])
#         return response
#     return inner


def x_shape_value_error_message(x, desired_dims):
    return f"""
    x provided is of shape {x.shape} but must have {len(desired_dims)}
    dimensions each representing: {desired_dims}. Note that in general,
    - N is the number of sample points to evaluate
    - q is the number of parallel points to evaluate (this dimension can be
      omitted for non-q acquisition functions)
    - d is the dimension of the input
    """


@define
class AcquisitionFunction(ABC, MSONable):
    bounds = field(validator=instance_of(tuple(DATA_TYPES)))
    q = field(default=1, validator=[instance_of(int), ge(1)])
    penalty_function = field(
        default=lambda _: 0.0, validator=instance_of(Callable)
    )
    penalty_strength = field(
        default=0.0, validator=[ge(0.0), instance_of((float, int))]
    )
    verbose = field(default=0, validator=[instance_of(int), ge(0)])
    batch_threshold = field(default=100, validator=[instance_of(int), gt(0)])

    def __attrs_post_init__(self):
        # The shape of bounds should always be two rows by as many columns
        # as there are dimensions
        self.bounds = jnp.array(self.bounds)
        assert self.bounds.ndim == 2
        assert self.bounds.shape[0] == 2

    def __call__(self, key, model, x):
        """When calling the acquisition function, it will evaluate the input
        x at every point provided. Generically, the input can be of arbitrary
        shape, and will be reshaped automatically. However, the last axis
        of the input must be equal to the input dimension of the GP model."""
        warn("This is a dummy __call__ instance!")

        x = jnp.array(x)
        if x.ndim != 2:
            raise ValueError(x_shape_value_error_message(x, ("N", "d")))
        mu, _ = model.predict(key, x)
        return mu

    def optimize_halton(self, key, model, n_samples=1000):
        """Optimizes the acquisition function using Halton random
        sampling/Monte Carlo."""

        halton = qmc.Halton(d=self.bounds.shape[1] * self.q, seed=key)
        samples = halton.random(n=n_samples)
        l_bounds = self.bounds[0, :].squeeze().tolist()
        u_bounds = self.bounds[1, :].squeeze().tolist()
        samples = qmc.scale(samples, l_bounds, u_bounds)
        samples = samples.reshape(n_samples, self.q, -1)
        samples_split = split_array(samples, s=self.batch_threshold)
        verbose = self.verbose > 0 and len(samples_split) > 1

        vals = []
        for xx in tqdm(samples_split, disable=not verbose):
            # Note that self.__call__ calls sample, and sample executes all
            # necessary transforms on the data
            vals.append(self(jra.key(key), model, xx))
        vals = jnp.array(vals).flatten()
        argmax = vals.argmax()

        return samples[argmax, ...], vals[argmax].item()

        if self.verbose > 0:
            quality = qmc.discrepancy(sample)
            print(f"qmc discrepancy (sample quality index) = {quality}")

    # def optimize(self, rng_key, n_initial=100):
    #     """Optimizes the provided acquisition function"""
    #
    #     shape = (n_initial, self.bounds.shape[1])
    #     minval = self.bounds[0, :]
    #     maxval = self.bounds[1, :]
    #     x0 = jra.uniform(rng_key, shape=shape, minval=minval, maxval=maxval)
    #
    #     def _negated_objective(x):
    #         # Returns the overall objective function negated
    #         # including the penalty term
    #         objective = self.__call__(rng_key, x) - self.penalty_function(x)
    #         return -objective
    #
    #     initial_acqf_vals = _negated_objective(x0)
    #     best_initial_guess = x0[initial_acqf_vals.argmax()].squeeze()
    #
    #     minimizer = jaxopt.ScipyBoundedMinimize(fun=)
    #
    #


@define
class qUpperConfidenceBound(AcquisitionFunction):
    beta = field(default=10.0, validator=[instance_of((int, float)), ge(0.0)])

    def __call__(self, rng_key, model, x):
        """UCB call function. x should be of shape (N, q, d), where N is the
        number of evaluation points to consider."""

        x = jnp.array(x)
        if x.ndim != 3 or x.shape[1] != self.q:
            raise ValueError(x_shape_value_error_message(x, ("N", "q", "d")))
        N = x.shape[0]
        d = x.shape[2]

        # samples is always of shape (hp_samples, gp_samples, N x q)
        samples = model.sample(rng_key, x.reshape(-1, d))
        samples = samples["y"].reshape(-1, N, self.q)
        mean = samples.mean(axis=0, keepdims=True)
        beta_prime = jnp.sqrt(self.beta * np.pi / 2.0)
        integrand = mean + beta_prime * jnp.abs(samples - mean)

        # Evaluate the max over the q axis, producing a
        # (hp_samples x gp_samples, N) array
        evaluate_max = integrand.max(axis=-1)

        # Return the average over samples of the GP
        return evaluate_max.mean(axis=0)
