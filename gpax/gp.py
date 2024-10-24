"""
gp.py
=======

- Fully Bayesian implementation of Gaussian process regression
- Variationa inference implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from warnings import warn

import jax
import jax.numpy as jnp
import jax.random as jra
import numpy as np
import numpyro
import numpyro.distributions as dist
from attrs import define, field, frozen
from attrs.validators import gt, instance_of
from monty.json import MSONable
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    Trace_ELBO,
    init_to_median,
)
from numpyro.infer.autoguide import AutoNormal

from gpax import state
from gpax.kernels import Kernel
from gpax.logger import logger
from gpax.transforms import IdentityTransform, ScaleTransform, Transform
from gpax.utils import time_function

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


DATA_TYPES = [jnp.ndarray, np.ndarray, type(None)]
Y_STD_DATA_TYPES = DATA_TYPES + [float] + [int]
DATA_TYPES = tuple(DATA_TYPES)
Y_STD_DATA_TYPES = tuple(Y_STD_DATA_TYPES)


@frozen
class GPSample(MSONable):
    """Container for accessing GP results."""

    _hp = field(default=None)
    _y = field(default=None)
    _mu = field(default=None)
    _sd = field(default=None)

    @property
    def y(self):
        """Samples over the GP. The shape of the returned array is
        (hp, gp, L), where n_hp indexes the kernel hyperparameter, gp indexes
        the sample from the GP, and L indexes the spatial component of the
        stochastic process."""

        return self._y

    @property
    def hp(self):
        """Samples of the hyperparameters."""

        return self._hp

    @cached_property
    def mu(self):
        """The mean of the GP."""

        if self._y.shape[0] == 1:
            return self._mu.squeeze()
        return np.mean(self._y, axis=(0, 1)).squeeze()

    @cached_property
    def sd(self):
        """The standard deviation of the GP."""

        if self._y.shape[0] == 1:
            return self._sd.squeeze()
        return np.std(self._y, axis=(0, 1)).squeeze()

    @cached_property
    def ci(self):
        """Confidence interval of mu +/- 2 sd."""

        return self.mu - 2.0 * self.sd, self.mu + 2.0 * self.sd


@define
class GaussianProcess(ABC, MSONable):
    """Core Gaussian process class.

    Parameters
    ----------
    hp_samples : int
        The number of samples to take over the distribution of hyper
        parameters. This also corresponds to the chain length in MCMC
        posterior sampling - in that case, this argument is ignored.
    gp_samples : int
        The number of samples over the GP to take for each sampled
        hyperparameter.
    """

    kernel = field(validator=instance_of(Kernel))
    x = field(default=None, validator=instance_of(DATA_TYPES))
    y = field(default=None, validator=instance_of(DATA_TYPES))
    y_std = field(default=None, validator=instance_of(Y_STD_DATA_TYPES))
    observation_noise = field(default=False, validator=instance_of(bool))
    hp_samples = field(default=100, validator=[instance_of(int), gt(0)])
    gp_samples = field(default=10, validator=[instance_of(int), gt(0)])
    input_transform = field(
        factory=ScaleTransform, validator=instance_of((Transform, type(None)))
    )
    output_transform = field(
        factory=IdentityTransform,
        validator=instance_of((Transform, type(None))),
    )
    verbose = field(default=0, validator=instance_of(int))
    metadata = field(factory=dict)
    use_cholesky = field(default=False, validator=instance_of(bool))
    _is_fit = field(default=False, validator=instance_of(bool))

    def _gp_prior(self, x, y=None):
        """The simple GP model. This is of course meant to be used with the
        appropriate numpyro primitives and in general not called directly.

        Parameters
        ----------
        x : array_like
            The input points.
        y : array_like, optional
            The optional points to condition the model on.
        """

        m = jnp.zeros(x.shape[0])
        k = self.kernel.sample_prior(x, x)
        return numpyro.sample(
            "y_sampled",
            dist.MultivariateNormal(loc=m, covariance_matrix=k),
            obs=y,
        )

    @abstractmethod
    def sample(self): ...

    @abstractmethod
    def predict(self): ...

    @abstractmethod
    def fit(self): ...

    def __attrs_pre_init__(self):
        clear_cache()

    def _class_specific_post_init(self):
        pass

    def _fit_transforms(self):
        # This works even if x and y are None
        self.input_transform.fit(self.x)
        self.output_transform.fit(self.y)

    def __attrs_post_init__(self):
        # Check the status of y_std
        if self.x is None and self.y_std is not None:
            raise ValueError("y_std cannot be set if no x or y data is given")

        # Assign y_std correcly if necessary
        if self.y is not None and isinstance(self.y_std, (float, int)):
            self.y_std = jnp.ones(self.y.shape) * self.y_std

        # Assign everything as Array objects to keep track of transformations
        if (self.y is None) ^ (self.x is None):
            raise ValueError("x and y must either both be None or not")

        # Set to identity transform in certain cases
        if self.input_transform is None or self.x is None:
            self.input_transform = IdentityTransform()
        if self.output_transform is None or self.y is None:
            self.output_transform = IdentityTransform()

        # Fit the transformation objects on all provided data
        self._fit_transforms()
        self._class_specific_post_init()

    @property
    def x_transformed(self):
        return self.input_transform.forward(self.x, transforms_as="mean")

    @property
    def y_transformed(self):
        y = self.output_transform.forward(self.y, transforms_as="mean")
        return y.squeeze()

    @property
    def y_std_transformed(self):
        y_std = self.output_transform.forward(self.y_std, transforms_as="std")
        if y_std is None:
            return None
        return y_std.squeeze()

    def _get_mean_and_covariance_unconditioned(self, x_new, kp):
        """A utility to get the multivariate normal prior given the GP mean
        function (which for now is just 0) and the pre-set kernel parameter
        priors.

        Parameters
        ----------
        x_new : jnp.ndarray
            A set of points to condition the GP on. As this is a "private"
            method, it is assumed that x_new is already transformed.

        Returns
        -------
        tuple
            Two jnp.ndarrays for the mean and covariance matrix, respectively.
        """

        # TODO: eventually reincorporate the mean function functionality
        mean = jnp.zeros(x_new.shape[0])

        f = self.kernel.kernel
        kp = deepcopy(kp)  # Kernel params

        k_noise = kp.pop("k_noise")
        k_jitter = kp.pop("k_jitter")

        if not self.observation_noise:
            noise = 0.0
        else:
            noise = k_noise
        cov = f(x_new, x_new, k_noise=noise, k_jitter=k_jitter, **kp)

        return mean, cov

    @staticmethod
    @time_function
    def _standard_condition(k_XX, k_pX, k_pp, y):
        k_XX_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(k_XX_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(k_XX_inv, y))
        return mean, cov

    @staticmethod
    @time_function
    def _cholesky_condition(k_XX, k_pX, k_pp, y):
        k_XX_cho = jax.scipy.linalg.cho_factor(k_XX)
        A = jax.scipy.linalg.cho_solve(k_XX_cho, k_pX.T)
        tt = jnp.matmul(k_pX, A)
        cov = k_pp - tt  # this is the bottleneck
        B = jax.scipy.linalg.cho_solve(k_XX_cho, y)
        mean = jnp.matmul(k_pX, B)
        return mean, cov

    @time_function
    def _get_mean_and_covariance(self, x_new, kp):
        """A utility to get the multivariate normal posterior given the GP
        and the training set of data to condition on.

        Parameters
        ----------
        x_new : jnp.ndarray
            A set of points to condition the GP on. As this is a "private"
            method, it is assumed that x_new is already transformed.
        kp : dict
            A dictionary containing the kernel hyperparameter samples.

        Returns
        -------
        tuple
            Two jnp.ndarrays for the mean and covariance matrix, respectively.
        """

        f = self.kernel.kernel
        kp = deepcopy(kp)  # Kernel params

        k_noise = kp.pop("k_noise")
        k_jitter = kp.pop("k_jitter")

        x = self.x_transformed
        y = self.y_transformed
        y_std = self.y_std_transformed

        k_pX = f(x_new, x, apply_noise=False, **kp)
        if not self.observation_noise:
            noise = 0.0
        else:
            noise = k_noise
        k_pp = f(x_new, x_new, k_noise=noise, k_jitter=k_jitter, **kp)

        if y_std is not None:
            noise = y_std**2
        else:
            noise = k_noise
        k_XX = f(x, x, k_noise=noise, k_jitter=k_jitter, **kp)

        if self.use_cholesky:
            mean, cov = self._cholesky_condition(k_XX, k_pX, k_pp, y)
        else:
            mean, cov = self._standard_condition(k_XX, k_pX, k_pp, y)
        return mean, cov

    def _sample_gp_given_single_hp(
        self, rng_key, x_new, kernel_params, n, condition_on_data=True
    ):
        """Execute random samples over the GP for an explicit set of kernel
        parameters. As this is a "private" method, it is assumed x_new is
        already transformed.

        Note that the key here is explicitly provided as this is a private
        method.
        """

        # Revert to the prior distribution if no updated kernel parameters
        # are provided
        kernel_params = {**self.kernel.kernel_params, **kernel_params}
        if condition_on_data:
            mean, cov = self._get_mean_and_covariance(x_new, kp=kernel_params)
        else:
            mean, cov = self._get_mean_and_covariance_unconditioned(
                x_new, kp=kernel_params
            )
        sampled = dist.MultivariateNormal(mean, cov).sample(
            rng_key, sample_shape=(n,)
        )
        return mean, jnp.sqrt(cov.diagonal()), sampled

    def _get_hp_samples_from_prior(self, rng_key, fast):
        f = self.kernel.sample_parameters
        hp_samples = Predictive(f, num_samples=self.hp_samples)(rng_key)
        n_hp_samples = self.hp_samples
        if fast:
            hp_samples = {
                key: jnp.median(value, keepdims=True)
                for key, value in hp_samples.items()
            }
            n_hp_samples = 1
        return hp_samples, n_hp_samples

    def _sample_prior(self, rng_key, x_new, condition_on_data, fast):
        hp_samples, n_hp = self._get_hp_samples_from_prior(rng_key, fast)
        predictive = jax.vmap(
            lambda p: self._sample_gp_given_single_hp(
                p[0],
                x_new,
                p[1],
                self.gp_samples,
                condition_on_data=condition_on_data,
            )
        )
        keys = jra.split(rng_key, n_hp)
        mu, sd, sampled = predictive((keys, hp_samples))
        return hp_samples, mu, sd, sampled

    def _sample_unconditioned_prior(self, rng_key, x_new, fast):
        """As this is a "private" method, it is assumed x_new is already
        transformed."""

        return self._sample_prior(rng_key, x_new, False, fast)

    def _sample_conditioned_prior(self, rng_key, x_new, fast):
        """As this is a "private" method, it is assumed x_new is already
        transformed."""

        return self._sample_prior(rng_key, x_new, True, fast)

    @abstractmethod
    def _get_hp_samples_from_posterior(self, fast): ...

    def _sample_posterior(self, rng_key, x_new, fast=True):
        hp_samples, n_hp = self._get_hp_samples_from_posterior(fast)
        predictive = jax.vmap(
            lambda p: self._sample_gp_given_single_hp(
                p[0], x_new, p[1], self.gp_samples, condition_on_data=True
            )
        )
        keys = jra.split(rng_key, n_hp)
        mu, sd, sampled = predictive((keys, hp_samples))
        return hp_samples, mu, sd, sampled

    def _pre_sample(self, x_new):
        x_new = jnp.array(x_new)
        _, rng_key = state.get_rng_key()
        x_new = jax.device_put(x_new, state.device)
        return rng_key, x_new

    def sample(self, x_new, fast=False):
        """Runs samples over the GP. These samples are some combination of
        results from sampling over the prior (or posterior) distribution of
        hyperparameters, and sampling from a GP assuming fixed hyperparameters.
        In some cases, hp_samples can be specified (such as when sampling
        from the unconditioned prior or the conditioned prior, without
        learning the parameters), but in the case of sampling over the
        posterior, it is fixed by the MCMC chain length.

        Notes
        -----
        Which distribution to sample over is specified by the overall form
        of the model and how it was initialized. If the data attribute is None,
        sample will draw samples from the prior distribution, which is usually
        just the zero mean function and the kernel paramters defined by the
        priors on the kernel. If data is not None and the model is not fit,
        samples are drawn from the conditioned prior. If data is not None and
        inference has been run via either MCMC or variational inference,
        then samples are drawn from the posterior.

        Parameters
        ----------
        x_new : array_like
            The input grid to sample on. Note that this is a "public" method
            and as such it is assumed x_new is _not_ transformed.
        fast : bool
            If True, will take the median of the sampled hyperparameters only
            instead of running predictions over every sample. This is
            particularly helpful for situations where each prediction is slow
            and we can tolerate a coarser approximation. Default is None
            (defaults to the value set at instantiation).

        Returns
        -------
        GPResults
            A class containing all of the samples over the hyperparameters
            and observations.
        """

        x_new = self.input_transform.forward(x_new, transforms_as="mean")

        rng_key, x_new = self._pre_sample(x_new)

        a = (rng_key, x_new, fast)
        if self.x is None and self.y is None:
            hp, mu, sd, y = self._sample_unconditioned_prior(*a)
        elif not self._is_fit:
            hp, mu, sd, y = self._sample_conditioned_prior(*a)
        else:
            hp, mu, sd, y = self._sample_posterior(*a)

        # Postprocess, send to cpu and convert to numpy arrays
        cpu = jax.devices("cpu")[0]
        y = np.array(jax.device_put(y, cpu))
        mu = np.array(jax.device_put(mu, cpu))
        sd = np.array(jax.device_put(sd, cpu))
        hp = {k: np.atleast_1d(jax.device_put(v, cpu)) for k, v in hp.items()}

        # Transform back
        y = self.output_transform.reverse(y, transforms_as="mean")
        mu = self.output_transform.reverse(mu, transforms_as="mean")
        sd = self.output_transform.reverse(sd, transforms_as="std")

        return GPSample(hp=hp, y=y, mu=mu, sd=sd)

    def _fast_predict(self, rng_key, x_new):
        """For an ExactGP, the fast prediction method will use only the
        median of the sampled hyperparameters. For a VIGP, only the median
        is used anyway."""

        fast = True
        if self.x is None and self.y is None:
            hps, _ = self._get_hp_samples_from_prior(rng_key, fast)
            condition_on_data = False
        elif not self._is_fit:
            hps, _ = self._get_hp_samples_from_prior(rng_key, fast)
            condition_on_data = True
        else:
            hps, _ = self._get_hp_samples_from_posterior(fast)
            condition_on_data = True

        if condition_on_data:
            mu, cov = self._get_mean_and_covariance(x_new, hps)
        else:
            mu, cov = self._get_mean_and_covariance_unconditioned(x_new, hps)
        return mu, jnp.sqrt(cov.diagonal())

    def predict(self, x_new, fast=False):
        if not fast:
            # Note that sample handles the transforms and whether or not
            # to sample from the prior or posterior, and whether or not to
            # condition on the data or not
            sampled = self.sample(x_new, False)
            return sampled.mu.squeeze(), sampled.sd.squeeze()
        # Otherwise, we need to access the particular method's fast_predict
        # method, whatever that may be!
        x_new = self.input_transform.forward(x_new, transforms_as="mean")
        rng_key, x_new = self._pre_sample(x_new)
        # underscore method assumes data is transformed coming in
        mu, sd = self._fast_predict(rng_key, x_new)
        mu = self.output_transform.reverse(mu, transforms_as="mean")
        sd = self.output_transform.reverse(sd, transforms_as="std")
        return mu.squeeze(), sd.squeeze()


@define
class ExactGP(GaussianProcess):
    mcmc = field(default=None)
    num_warmup = field(default=2000, validator=[instance_of(int), gt(0)])
    num_chains = field(default=1, validator=[instance_of(int), gt(0)])
    chain_method = field(default="sequential", validator=instance_of(str))
    mcmc_run_kwargs = field(factory=dict)

    def fit(self):
        """Runs Hamiltonian Monte Carlo to infer the GP parameters."""

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self._gp_prior, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.hp_samples,
            num_chains=self.num_chains,
            chain_method=self.chain_method,
            progress_bar=self.verbose > 0,
            jit_model_args=False,
        )
        x, y = self.x_transformed, self.y_transformed
        x = jax.device_put(x, state.device)
        y = jax.device_put(y, state.device)
        key, rng_key = state.get_rng_key()
        self.mcmc.run(rng_key, x, y, **self.mcmc_run_kwargs)
        if self.verbose > 0:
            self.mcmc.print_summary()
        self.metadata["fit_key"] = np.array(key)
        self._is_fit = True

    def _get_hp_samples_from_posterior(self, fast):
        hp_samples = self.mcmc.get_samples(group_by_chain=False)
        if fast:
            hp_samples = {
                key: jnp.median(value, keepdims=True)
                for key, value in hp_samples.items()
            }
        n_hp_samples = len(next(iter(hp_samples.values())))
        return hp_samples, n_hp_samples


@define
class VariationalInferenceGP(GaussianProcess):
    guide_factory = field(default=AutoNormal)
    svi = field(default=None)
    loss_factory = field(default=Trace_ELBO)
    kernel_params = field(default=None)
    num_steps = field(default=100, validator=[instance_of(int), gt(0)])
    optimizer_factory = field(default=numpyro.optim.Adam)
    optimizer_kwargs = field(factory=dict)
    hp_samples = field(default=1, validator=[instance_of(int), gt(0)])

    def _print_summary(self):
        params_map = self.svi.guide.median(self.kernel_params.params)
        logger.info("\nInferred GP parameters")
        for k, vals in params_map.items():
            spaces = " " * (15 - len(k))
            logger.info(k, spaces, jnp.around(vals, 4))

    def _class_specific_post_init(self):
        if "b1" not in self.optimizer_kwargs:
            self.optimizer_kwargs["b1"] = 0.5
        if "step_size" not in self.optimizer_kwargs:
            self.optimizer_kwargs["step_size"] = 1e-3
        if self.hp_samples != 1:
            warn(
                f"hp_samples is set to {self.hp_samples} but will be "
                "overridden to 1. In VariationalInferenceGP, only the "
                "median value of the hyperparameters are used for predictions."
            )
            self.hp_samples = 1

    def fit(self):
        _, rng_key = state.get_rng_key()
        optim = self.optimizer_factory(**self.optimizer_kwargs)
        guide = self.guide_factory(self._gp_prior)
        loss = self.loss_factory()
        self.svi = SVI(self._gp_prior, guide=guide, optim=optim, loss=loss)
        x, y = self.x_transformed, self.y_transformed
        self.kernel_params = self.svi.run(
            rng_key, self.num_steps, progress_bar=self.verbose > 0, x=x, y=y
        )
        if self.verbose > 0:
            self._print_summary()
        self.metadata["fit_key"] = np.array(rng_key)
        self._is_fit = True

    def _get_hp_samples_from_posterior(self, fast):
        if fast:
            logger.warning(
                "Note setting fast = True for VIGP will do nothing."
                "VIGP only uses the median value of the hyperparameters by "
                "default."
            )
        kernel_params = self.svi.guide.median(self.kernel_params.params)
        kernel_params = {
            key: jnp.atleast_1d(value) for key, value in kernel_params.items()
        }
        return kernel_params, 1
