"""
gp.py
=======

Fully Bayesian implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import numpyro
import numpyro.distributions as dist
from attrs import define, field
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

from gpax.data import Array, NormalizeTransform, ScaleTransform, Transform
from gpax.kernels import Kernel

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


DATA_TYPES = [jnp.ndarray, np.ndarray, type(None)]
Y_STD_DATA_TYPES = DATA_TYPES + [float] + [int]
DATA_TYPES = tuple(DATA_TYPES)
Y_STD_DATA_TYPES = tuple(Y_STD_DATA_TYPES)


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
        factory=ScaleTransform, validator=instance_of(Transform)
    )
    output_transform = field(
        factory=NormalizeTransform, validator=instance_of(Transform)
    )
    _is_fit = field(default=False, validator=instance_of(bool))

    def _model(self, x, y=None):
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
            "y",
            dist.MultivariateNormal(loc=m, covariance_matrix=k),
            obs=y,
        )

    @abstractmethod
    def sample(self): ...

    @abstractmethod
    def fit(self): ...

    def __attrs_pre_init__(self):
        clear_cache()

    def _assign_arrays(self):
        if self.x is not None:
            self.x = Array(self.x)
            self.x.metadata.transforms_as = "mean"
        if self.y is not None:
            self.y = Array(self.y)
            self.y.metadata.transforms_as = "mean"
        if self.y_std is not None:
            self.y_std = Array(self.y_std)
            self.y_std.metadata.transforms_as = "std"
        if (self.y is None) ^ (self.x is None):
            raise ValueError("x and y must either both be None or not")

    def _fit_transforms(self):
        # This works even if x and y are None
        self.input_transform.fit(self.x)
        self.output_transform.fit(self.y)

    def __attrs_post_init__(self):
        if isinstance(self.y_std, (float, int)):
            self.y_std = jnp.ones(self.y_std.shape) * self.y_std

        # Assign everything as Array objects to keep track of transformations
        self._assign_arrays()

        # Fit the transformation objects on all provided data
        self._fit_transforms()

    def _get_transformed_data(self):
        x = self.input_transform.forward(self.x)
        y = self.output_transform.forward(self.y)
        if y is not None:
            y = y.squeeze()
        y_std = self.output_transform.forward(self.y_std)
        if y_std is not None:
            y_std = y_std.squeeze()
        return x, y, y_std

    def _get_untransformed_data(self):
        x = self.input_transform.reverse(self.x)
        y = self.output_transform.reverse(self.y)
        if y is not None:
            y = y.squeeze()
        y_std = self.output_transform.reverse(self.y_std)
        if y_std is not None:
            y_std = y_std.squeeze()
        return x, y, y_std

    def _forward_transform_input_as_mean(self, x):
        x = Array(x)
        x.metadata.transforms_as = "mean"
        return self.input_transform.forward(x)

    def _get_mvn(self, x_new, kp):
        """A utility to get the multivariate normal posterior given the GP
        and a new set of data to condition on.

        Parameters
        ----------
        x_new : jnp.ndarray
            A set of points to condition the GP on.

        Returns
        -------
        tuple
            Two jnp.ndarrays for the mean and covariance matrix, respectively.
        """

        x_new = self._forward_transform_input_as_mean(x_new)
        x, y, y_std = self._get_transformed_data()

        f = self.kernel.kernel
        kp = deepcopy(kp)  # Kernel params

        k_noise = kp.pop("k_noise")
        k_jitter = kp.pop("k_jitter")

        k_pX = f(x_new, x, apply_noise=False, **kp)
        if not self.observation_noise:
            noise = self.observation_noise
        else:
            noise = k_noise
        k_pp = f(x_new, x_new, k_noise=noise, k_jitter=k_jitter, **kp)

        if y_std is not None:
            noise = y_std**2
        else:
            noise = k_noise
        k_XX = f(x, x, k_noise=noise, k_jitter=k_jitter, **kp)

        y_residual = y.copy()
        k_XX_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(k_XX_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(k_XX_inv, y_residual))
        return mean, cov

    def _sample(self, rng_key, x_new, kernel_params, n):
        """Execute random samples over the GP for an explicit set of kernel
        parameters."""

        x_new = self._forward_transform_input_as_mean(x_new)
        kernel_params = {**self.kernel.kernel_params, **kernel_params}
        mean, cov = self._get_mvn(x_new, kp=kernel_params)
        sampled = dist.MultivariateNormal(mean, cov).sample(
            rng_key, sample_shape=(n,)
        )
        return sampled

    def _sample_unconditioned_prior(self, rng_key, x_new):
        x_new = self._forward_transform_input_as_mean(x_new)
        gp_samples = self.gp_samples
        samples = Predictive(self._model, num_samples=gp_samples)(
            rng_key, x_new
        )
        n = samples["y"].shape[-1]
        samples["y"] = samples["y"].reshape(-1, 1, n)
        return samples

    def _sample_conditioned_prior(self, rng_key, x_new):
        x_new = self._forward_transform_input_as_mean(x_new)
        f = self.kernel.sample_parameters
        samples = Predictive(f, num_samples=self.hp_samples)(rng_key)
        predictive = jax.vmap(
            lambda p: self._sample(p[0], x_new, p[1], self.gp_samples)
        )
        keys = jrp.split(rng_key, self.hp_samples)
        sampled = predictive((keys, samples))
        return {**samples, "y": sampled}

    @abstractmethod
    def _sample_posterior(self): ...

    def sample(self, rng_key, x_new):
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
        rng_key : int
            Key for seeding the random number generator.
        x_new : array_like
            The input grid to sample on.

        Returns
        -------
        dict
            A dictionary containing all of the samples over the hyperparameters
            and observations. The observations in particular are of the shape
            (hp_samples, gp_samples, X_new.shape[0]) array corresponding to the
            sampled results.
        """

        x_new = self._forward_transform_input_as_mean(x_new)

        if self.x is None and self.y is None:
            samples = self._sample_unconditioned_prior(rng_key, x_new)
        elif not self._is_fit:
            samples = self._sample_conditioned_prior(rng_key, x_new)
        else:
            samples = self._sample_posterior(rng_key, x_new)

        # Inverse transform the samples
        y = samples["y"]
        original_y_shape = y.shape
        y = y.reshape(-1, y.shape[-1])

        y = Array(y)
        y.metadata.is_transformed = True
        y.metadata.transforms_as = "mean"
        y = self.output_transform.reverse(y).reshape(*original_y_shape)

        samples["y"] = np.array(y)

        return samples

    def predict(self, rng_key, x_new):
        """Finds the mean and variance of the model via sampling.

        Parameters
        ----------
        rng_key : int
            Key for seeding the random number generator.
        x_new : array_like
            The input grid to find the mean and variance predictions on.

        Returns
        -------
        Two arrays, one for the mean, and the other for the variance of the
        predictions evaluated on the grid x_new.
        """

        x_new = self._forward_transform_input_as_mean(x_new)
        samples = self.sample(rng_key, x_new)
        y = samples["y"]
        mean, std = jnp.mean(y, axis=[0, 1]), jnp.std(y, axis=[0, 1])

        mean = Array(mean)
        mean.metadata.transforms_as = "mean"
        mean = self.output_transform.reverse(mean).squeeze()

        std = Array(std)
        std.metadata.transforms_as = "std"
        std = self.output_transform.reverse(std).squeeze()

        return mean, std


@define
class ExactGP(GaussianProcess):
    mcmc = field(default=None)

    def fit(
        self,
        rng_key,
        num_warmup=2000,
        num_samples=2000,
        num_chains=1,
        chain_method="sequential",
        progress_bar=True,
        print_summary=True,
        **mcmc_run_kwargs: float,
    ):
        """Runs Hamiltonian Monte Carlo to infer the GP parameters.

        Parameters
        ----------
        rng_key : int
            Random number generator key.


        Args:
            rng_key: random number generator key
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]``
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)
        """

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self._model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        x, y, _ = self._get_transformed_data()
        self.mcmc.run(rng_key, x, y, **mcmc_run_kwargs)
        if print_summary:
            self.mcmc.print_summary()
        self._is_fit = True

    def _sample_posterior(self, rng_key, x_new):
        x_new = self._forward_transform_input_as_mean(x_new)
        samples = self.mcmc.get_samples(group_by_chain=False)
        chain_length = len(next(iter(samples.values())))
        predictive = jax.vmap(
            lambda p: self._sample(p[0], x_new, p[1], self.gp_samples)
        )
        keys = jrp.split(rng_key, chain_length)
        sampled = predictive((keys, samples))
        return {**samples, "y": sampled}


@define
class VariationalInferenceGP(GaussianProcess):
    guide_factory = field(default=AutoNormal)
    svi = field(default=None)
    optimizer_factory = field(
        default=partial(numpyro.optim.Adam, step_size=1e-3, b1=0.5)
    )
    loss_factory = field(default=Trace_ELBO)
    kernel_params = field(default=None)

    def _print_summary(self):
        params_map = self.svi.guide.median(self.kernel_params.params)
        print("\nInferred GP parameters")
        for k, vals in params_map.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))

    def fit(self, rng_key, num_steps, progress_bar=True, print_summary=True):
        optim = self.optimizer_factory()
        guide = self.guide_factory(self._model)
        loss = self.loss_factory()
        self = SVI(
            self._model,
            guide=guide,
            optim=optim,
            loss=loss,
        )
        x, y, _ = self._get_transformed_data()
        self.kernel_params = self.svi.run(
            rng_key, num_steps, progress_bar=progress_bar, x=x, y=y
        )
        if print_summary:
            self._print_summary()
        self._is_fit = True

    def _sample_posterior(self, rng_key, x_new):
        x_new = self._forward_transform_input_as_mean(x_new)
        kernel_params = self.svi.guide.median(self.kernel_params.params)
        samples = self._sample(rng_key, x_new, kernel_params, self.gp_samples)
        samples = samples.reshape(1, samples.shape[0], samples.shape[1])
        kernel_params["y"] = samples
        return kernel_params

    def predict(self, rng_key, x_new):
        # Override the default sampling behavior if the model is fit and
        # the data is provided. SVI is special in that there is only the
        # median kernel parameters to consider

        x_new = self._forward_transform_input_as_mean(x_new)

        if self._is_fit:
            kernel_params = self.svi.guide.median(self.kernel_params.params)
            mean, cov = self._get_mvn(x_new, kernel_params)
            return mean, cov.diagonal()

        return super().sample(rng_key, x_new)
