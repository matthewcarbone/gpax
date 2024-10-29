"""
acquisition.py
==============

Base acquisition functions and objects

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp

# import jaxopt
import numpy as np
import numpyro.distributions as dist
from attrs import define, field
from attrs.validators import ge, gt, instance_of
from monty.json import MSONable
from scipy.stats import qmc
from tqdm import tqdm

from gpax import state
from gpax.logger import logger
from gpax.utils import split_array

DATA_TYPES = [jnp.ndarray, np.ndarray]


def x_shape_err_msg(x, desired_dims):
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
    force_monte_carlo = field(default=False, validator=instance_of(bool))
    fast = field(default=True, validator=instance_of(bool))

    def __attrs_post_init__(self):
        # The shape of bounds should always be two rows by as many columns
        # as there are dimensions
        self.bounds = jnp.array(self.bounds)
        assert self.bounds.ndim == 2
        assert self.bounds.shape[0] == 2

    def _get_integrand_samples(self, model, x):
        """Gets samples from the integrand given an array of shape (N, q, d)
        and returns only the observation values in the shape
        (hp_samples x gp_samples, N, q)."""

        N = x.shape[0]
        d = x.shape[2]

        # samples is always of shape (hp_samples, gp_samples, N x q)
        result = model.sample(x.reshape(-1, d), fast=self.fast)
        result = result.y.reshape(-1, N, self.q)
        return result

    @abstractmethod
    def _integrand(self, model, x):
        """The integrand of the acquisition function in integral form. See
        Wilson et al. (https://arxiv.org/abs/1712.00424). Specifically, this
        is the vale for h in Eq. (1). Note that this is only used and required
        for Monte Carlo acquisition (q>1). The integrand should return a
        shape of (N_MC_samples x )"""

        raise NotImplementedError

    def _analytic(self, model, x):
        """Returns the analytic form of the acquisition function evaluated at
        point x. Note that by default, this will raise a NotImplementedError,
        and the analytic form of the acquisition function, for q=1 only, must
        be defined in a child class. Note that since this method is only used
        for q=1, the input is expected to be of shape (N, d), not (N, 1, d).
        """

        raise NotImplementedError(
            f"Acquisition function {self.__class__.__name__} does not have an "
            "analytic acquisition function defined"
        )

    def _monte_carlo_evaluation(self, model, x):
        """Executes the evaluation of the monte carlo sampling procedure given
        design point x."""

        if x.ndim == 2 and self.q == 1:
            x = x.reshape(x.shape[0], 1, x.shape[-1])

        if x.ndim != 3 or x.shape[1] != self.q:
            raise ValueError(x_shape_err_msg(x, ("N", "q", "d")))

        # Get the value of the integrand, h, evaluated for many samples of
        # the GP
        h = self._integrand(model, x)

        # Return the average over samples of the GP
        return h.mean(axis=0)

    def _analytic_evaluation(self, model, x):
        if self.q > 1:
            raise ValueError("Cannot evaluate analytic expression for q!=1")

        if x.ndim == 3 and x.shape[1] == 1:
            x = x.reshape(x.shape[0], x.shape[-1])

        if x.ndim != 2:
            raise ValueError(x_shape_err_msg(x, ("N", "d")))

        return self._analytic(model, x)

    def __call__(self, model, x):
        """Produces the value of the acquisition function evaluated at point
        x.

        Parameters
        ----------
        model : gpax.gp.GaussianProcess
            A GaussianProcess instance.
        x : array_like
            The input array specifying the points to evaluate the acquisition
            function at. Generally, x should be of shape (N, q, d). If an
            input is provided with x.ndim == 3, this will be the assumption.
            If an input is provided with x.ndim == 2 and q == 1, it will be
            reshaped to (N, 1, d) if force_monte_carlo is set to True. If
            x.ndim == 2 and q > 1, an error will be raised.
                In the case where q == 1, and an analytic expression for the
            acquisition function is available, the expected input shape is
            (N, d). However, if an input of shape (N, 1, d) is provided, it
            is valid and will be reshaped accordingly.

        Returns
        -------
        An array of shape N, consisting of the values of the acquisition
        function at the provided points.
        """

        x = jnp.array(x)

        if self.force_monte_carlo or self.q > 1:
            return self._monte_carlo_evaluation(model, x)

        return self._analytic_evaluation(model, x)

    def _optimize_halton(self, model, n_samples):
        """Optimizes the acquisition function using Halton random
        sampling/Monte Carlo.

        Parameters
        ----------
        model : gpax.gp.GaussianProcess
            A GaussianProcess instance.
        n_samples : int
            The number of random Halton samples to draw during the optimization.
            Precisely, it's actually n_samples x q.

        Returns
        -------
        The q x d array containing the optimal sampled points, as well as the
        value of the acquisition function at the sampled point.
        """

        key, _ = state.get_rng_key()

        halton = qmc.Halton(d=self.bounds.shape[1] * self.q, seed=key)
        samples = halton.random(n=n_samples)

        # Need to retile the bounds so that they match the d * q dimension
        l_bounds = self.bounds[0, :].squeeze()
        l_bounds = np.tile(l_bounds, self.q)
        u_bounds = self.bounds[1, :].squeeze()
        u_bounds = np.tile(u_bounds, self.q)

        if self.verbose > 0:
            quality = qmc.discrepancy(samples)
            logger.debug(
                f"qmc discrepancy (sample quality index) = {quality:.02e}"
            )

        samples = qmc.scale(samples, l_bounds, u_bounds)
        samples = samples.reshape(n_samples, self.q, -1)
        # for _d in range(self.bounds.shape[1]):
        #     assert np.all(l_bounds[_d] <= samples[:, :, _d])
        samples_split = split_array(samples, s=self.batch_threshold)
        verbose = self.verbose > 0 and len(samples_split) > 1

        vals = []
        for xx in tqdm(samples_split, disable=not verbose):
            # Note that self.__call__ calls sample, and sample executes all
            # necessary transforms on the data
            vals.append(self.__call__(model, xx))
        vals = jnp.array(vals).flatten()
        argmax = vals.argmax()

        return samples[argmax, ...], vals[argmax].item()

    def optimize(self, model, n, method="halton"):
        method = method.lower()
        if method == "halton":
            return self._optimize_halton(model, n)
        else:
            raise ValueError(f"Unknown optimizer {method}")

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
class UpperConfidenceBound(AcquisitionFunction):
    """The upper confidence bound acquisition function (UCB). UCB strikes a
    balance between exploration and exploitation via parameter beta. Note that
    if beta == inf, UCB reduces to the MaximumVariance acquisition function."""

    beta = field(default=10.0, validator=[instance_of((int, float)), ge(0.0)])

    def _analytic(self, model, x):
        mu, sd = model.predict(x, fast=self.fast)
        if bool(jnp.isinf(self.beta)):
            return sd
        if self.beta == 0.0:
            return mu
        return mu + jnp.sqrt(self.beta + 1e-12) * sd

    def _integrand(self, model, x):
        samples = self._get_integrand_samples(model, x)
        mean = samples.mean(axis=0, keepdims=True)
        if bool(jnp.isinf(self.beta)):
            return jnp.abs(samples - mean).max(axis=-1)
        beta_prime = jnp.sqrt(self.beta * np.pi / 2.0)
        return (mean + beta_prime * jnp.abs(samples - mean)).max(axis=-1)


@define
class ExpectedImprovement(AcquisitionFunction):
    def _get_y_max(self, model):
        if model.y is not None:
            return model.y.max()

        # This is effectively finding the maximum of the prior
        acqf = UpperConfidenceBound(
            beta=0.0, q=1, bounds=self.bounds, fast=True
        )
        x_star, _ = acqf.optimize(model, n=1000, method="Halton")
        y_max, _ = model.predict(x_star, fast=True)
        return y_max

    def _analytic(self, model, x):
        mu, sd = model.predict(x, fast=self.fast)
        y_max = self._get_y_max(model)
        u = (mu - y_max) / sd
        normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
        ucdf = normal.cdf(u)
        updf = jnp.exp(normal.log_prob(u))
        return sd * updf + (mu - y_max) * ucdf

    def _integrand(self, model, x):
        samples = self._get_integrand_samples(model, x)
        y_max = self._get_y_max(model)
        f_max_over_q = samples.max(axis=-1)
        where_gt_0 = f_max_over_q > 0
        return (f_max_over_q - y_max) * where_gt_0


# @define
# class ProbabilityOfImprovement(AcquisitionFunction):
#     def analytic(self, key, model, x):
#         if best_f is None:
#             best_f = mean.max()
#         sigma = jnp.sqrt(variance)
#         u = (mean - y_max - xi) / sigma
#         normal = dist.Normal(jnp.zeros_like(u), jnp.ones_like(u))
#         return normal.cdf(u)
#
