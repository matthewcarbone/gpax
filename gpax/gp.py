"""
gp.py
=======

Fully Bayesian implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from warnings import warn

import jax
import jax.numpy as jnp
import jax.random as jra
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

from gpax import state
from gpax.kernels import Kernel
from gpax.transforms import IdentityTransform, ScaleTransform, Transform

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
        factory=ScaleTransform, validator=instance_of((Transform, type(None)))
    )
    output_transform = field(
        factory=IdentityTransform,
        validator=instance_of((Transform, type(None))),
    )
    verbose = field(default=0, validator=instance_of(int))
    metadata = field(factory=dict)
    use_cholesky = field(default=True, validator=instance_of(bool))
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
            "y",
            dist.MultivariateNormal(loc=m, covariance_matrix=k),
            obs=y,
        )

    @abstractmethod
    def sample(self): ...

    @abstractmethod
    def fit(self, key): ...

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
        y_std = self.input_transform.forward(self.y_std, transforms_as="std")
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
            noise = self.observation_noise
        else:
            noise = k_noise
        cov = f(x_new, x_new, k_noise=noise, k_jitter=k_jitter, **kp)

        return mean, cov

    @staticmethod
    def _standard_condition(k_XX, k_pX, k_pp, y):
        k_XX_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(k_XX_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(k_XX_inv, y))
        return mean, cov

    @staticmethod
    def _cholesky_condition(k_XX, k_pX, k_pp, y):
        k_XX_cho = jax.scipy.linalg.cho_factor(k_XX)
        A = jax.scipy.linalg.cho_solve(k_XX_cho, k_pX.T)
        cov = k_pp - jnp.matmul(k_pX, A)
        B = jax.scipy.linalg.cho_solve(k_XX_cho, y)
        mean = jnp.matmul(k_pX, B)
        return mean, cov

    def _get_mean_and_covariance(self, x_new, kp):
        """A utility to get the multivariate normal posterior given the GP
        and the training set of data to condition on.

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

        f = self.kernel.kernel
        kp = deepcopy(kp)  # Kernel params

        k_noise = kp.pop("k_noise")
        k_jitter = kp.pop("k_jitter")

        x = self.x_transformed
        y = self.y_transformed
        y_std = self.y_std_transformed

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

        if self.use_cholesky:
            mean, cov = self._cholesky_condition(k_XX, k_pX, k_pp, y)
        else:
            mean, cov = self._standard_condition(k_XX, k_pX, k_pp, y)
        return mean, cov

    def _sample(self, rng_key, x_new, kernel_params, n, condition_on_data=True):
        """Execute random samples over the GP for an explicit set of kernel
        parameters. As this is a "private" method, it is assumed x_new is
        already transformed."""

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
        return sampled

    def _sample_prior(self, rng_key, x_new, condition_on_data):
        f = self.kernel.sample_parameters
        hp_samples = Predictive(f, num_samples=self.hp_samples)(rng_key)
        predictive = jax.vmap(
            lambda p: self._sample(
                p[0],
                x_new,
                p[1],
                self.gp_samples,
                condition_on_data=condition_on_data,
            )
        )
        keys = jra.split(rng_key, self.hp_samples)
        sampled = jnp.array(predictive((keys, hp_samples)))
        return {**hp_samples, "y": sampled}

    def _sample_unconditioned_prior(self, rng_key, x_new):
        """As this is a "private" method, it is assumed x_new is already
        transformed."""

        return self._sample_prior(rng_key, x_new, False)

        # gp_samples = self.gp_samples
        # samples = Predictive(self._gp_prior, num_samples=gp_samples)(
        #     rng_key, x_new
        # )
        # n = samples["y"].shape[-1]
        # samples["y"] = samples["y"].reshape(-1, 1, n)
        # return samples

    def _sample_conditioned_prior(self, rng_key, x_new):
        """As this is a "private" method, it is assumed x_new is already
        transformed."""

        return self._sample_prior(rng_key, x_new, True)

        # f = self.kernel.sample_parameters
        # hp_samples = Predictive(f, num_samples=self.hp_samples)(rng_key)
        # predictive = jax.vmap(
        #     lambda p: self._sample(p[0], x_new, p[1], self.gp_samples)
        # )
        # keys = jra.split(rng_key, self.hp_samples)
        # sampled = jnp.array(predictive((keys, hp_samples)))
        # return {**hp_samples, "y": sampled}

    @abstractmethod
    def _sample_posterior(self): ...

    def sample(self, key, x_new):
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
        key : int
            Key for seeding the random number generator.
        x_new : array_like
            The input grid to sample on. Note that this is a "public" method
            and as such it is assumed x_new is _not_ transformed.

        Returns
        -------
        dict
            A dictionary containing all of the samples over the hyperparameters
            and observations. The observations in particular are of the shape
            (hp_samples, gp_samples, X_new.shape[0]) array corresponding to the
            sampled results. The resulting values for "y" will be reverse
            transformed back to the original space, but the values for the
            kernel parameters will not be.
        """

        if isinstance(key, int):
            rng_key = jra.key(key)
        else:
            rng_key = key

        x_new = self.input_transform.forward(x_new, transforms_as="mean")
        x_new = jax.device_put(x_new, state.device)

        if self.x is None and self.y is None:
            samples = self._sample_unconditioned_prior(rng_key, x_new)
        elif not self._is_fit:
            samples = self._sample_conditioned_prior(rng_key, x_new)
        else:
            samples = self._sample_posterior(rng_key, x_new)

        # Inverse transform the samples
        y = samples["y"]
        y = self.output_transform.reverse(y, transforms_as="mean")
        y = jnp.array(y)
        y = jax.device_put(y, jax.devices("cpu")[0])
        samples["y"] = y

        return samples

    def predict(self, key, x_new):
        """Finds the mean and variance of the model via sampling.

        Parameters
        ----------
        key : int
            Key for seeding the random number generator.
        x_new : array_like
            The input grid to find the mean and variance predictions on. As
            this is a "public" method, x_new should _not_ be transformed.

        Returns
        -------
        Two arrays, one for the mean, and the other for the variance of the
        predictions evaluated on the grid x_new.
        """

        rng_key = jra.key(key)

        # Note that samples here takes care of the transformations
        # samples actualy returns transformed results already
        samples = self.sample(rng_key, x_new)
        y = samples["y"]
        return jnp.mean(y, axis=[0, 1]), jnp.std(y, axis=[0, 1])


@define
class ExactGP(GaussianProcess):
    mcmc = field(default=None)
    num_warmup = field(default=2000, validator=[instance_of(int), gt(0)])
    num_chains = field(default=1, validator=[instance_of(int), gt(0)])
    chain_method = field(default="sequential", validator=instance_of(str))
    mcmc_run_kwargs = field(factory=dict)

    def fit(self, key):
        """Runs Hamiltonian Monte Carlo to infer the GP parameters.

        Parameters
        ----------
        key : int
            Random number generator key.
        """

        rng_key = jra.key(key)

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
        self.mcmc.run(rng_key, x, y, **self.mcmc_run_kwargs)
        if self.verbose > 0:
            self.mcmc.print_summary()
        self.metadata["fit_key"] = np.array(rng_key)
        self._is_fit = True

    def _sample_posterior(self, rng_key, x_new):
        samples = self.mcmc.get_samples(group_by_chain=False)
        chain_length = len(next(iter(samples.values())))
        predictive = jax.vmap(
            lambda p: self._sample(p[0], x_new, p[1], self.gp_samples)
        )
        keys = jra.split(rng_key, chain_length)
        sampled = predictive((keys, samples))
        return {**samples, "y": sampled}


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
        print("\nInferred GP parameters")
        for k, vals in params_map.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))

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

    def fit(self, key):
        rng_key = jra.key(key)
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

    def _sample_posterior(self, rng_key, x_new):
        kernel_params = self.svi.guide.median(self.kernel_params.params)
        samples = self._sample(rng_key, x_new, kernel_params, self.gp_samples)
        samples = samples.reshape(1, samples.shape[0], samples.shape[1])
        kernel_params["y"] = samples
        return kernel_params

    def predict(self, key, x_new):
        # Override the default sampling behavior if the model is fit and
        # the data is provided. SVI is special in that there is only the
        # median kernel parameters to consider

        if not self._is_fit:
            # sample will transform x_new for us as well as the samples
            # themselves
            sampled = super().sample(key, x_new)
            y = sampled["y"]
            return y.mean(axis=[0, 1]), y.std(axis=[0, 1])

        if key is not None:
            warn(
                "Unless the model is not fit, a random number is not "
                "used during predict for VariationalInferenceGP, as only "
                "the median value of the kernels are used for prediction. "
                "Provided key will be ignored."
            )

        # here the transforms need to be applied
        x_new = self.input_transform.forward(x_new, transforms_as="mean")
        kernel_params = self.svi.guide.median(self.kernel_params.params)
        mean, cov = self._get_mean_and_covariance(x_new, kernel_params)
        std = jnp.sqrt(cov.diagonal())
        mean = self.output_transform.reverse(mean, transforms_as="mean")
        std = self.output_transform.reverse(std, transforms_as="std")
        return mean.squeeze(), std.squeeze()
