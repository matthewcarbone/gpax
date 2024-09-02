"""
gp.py
=======

Fully Bayesian implementation of Gaussian process regression

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrp
import numpyro
import numpyro.distributions as dist
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

clear_cache = jax._src.dispatch.xla_primitive_callable.cache_clear


class Data(MSONable):
    def _assert_shapes(self):
        assert self.x.ndim == 2
        assert self.y.ndim == 1
        assert self.x.shape[0] == len(self.y)
        if self.y_var is not None:
            assert self.y_var.ndim == 1
            assert self.x.shape[0] == len(self.y_var)

    def __init__(self, x, y, y_var=None):
        self.x = x
        self.y = y
        if isinstance(y_var, (int, float)):
            self.y_var = jnp.ones(len(y)) * y_var
        else:
            self.y_var = y_var
        self._assert_shapes()


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

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input to the GP.
    kernel
    mean_function
    kernel_prior : dict
        A dictionary containing the prior distributions over the kernel
        parameters. For example:

        .. code-block:: python

            def gp_kernel_custom_prior():
                length = numpyro.sample(
                    "k_length", numpyro.distributions.Uniform(0, 1)
                )
                scale = numpyro.sample(
                    "k_scale", numpyro.distributions.LogNormal(0, 1)
                )
                return {"k_length": length, "k_scale": scale}

    """

    @abstractmethod
    def sample(self): ...

    @abstractmethod
    def fit(self): ...

    @property
    def x(self):
        return self.data.x

    @property
    def y(self):
        return self.data.y

    @property
    def y_var(self):
        return self.data.y_var

    def __init__(
        self,
        kernel,
        data,
        observation_noise,
        model_fit=False,
        hp_samples=100,
        gp_samples=10,
    ):
        clear_cache()
        self.kernel = kernel
        self.data = data
        self.observation_noise = observation_noise
        self.model_fit = model_fit
        self.hp_samples = hp_samples
        self.gp_samples = gp_samples

    def get_mvn(self, x_new, kp):
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

        f = self.kernel.kernel
        kp = deepcopy(kp)  # Kernel params

        k_noise = kp.pop("k_noise")
        k_jitter = kp.pop("k_jitter")

        k_pX = f(x_new, self.x, apply_noise=False, **kp)
        if not self.observation_noise:
            noise = self.observation_noise
        else:
            noise = k_noise
        k_pp = f(x_new, x_new, k_noise=noise, k_jitter=k_jitter, **kp)

        if self.y_var is not None:
            noise = self.y_var
        else:
            noise = k_noise
        k_XX = f(self.x, self.x, k_noise=noise, k_jitter=k_jitter, **kp)

        y_residual = self.y.copy()
        k_XX_inv = jnp.linalg.inv(k_XX)
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(k_XX_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(k_XX_inv, y_residual))
        return mean, cov

    def model(self, x, y=None):
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

    def _sample(self, rng_key, X_new, kernel_params, n):
        """Execute random samples over the GP for an explicit set of kernel
        parameters."""
        kernel_params = {**self.kernel.kernel_params, **kernel_params}
        mean, cov = self.get_mvn(X_new, kp=kernel_params)
        sampled = dist.MultivariateNormal(mean, cov).sample(
            rng_key, sample_shape=(n,)
        )
        return sampled

    def _sample_unconditioned_prior(self, rng_key, X_new):
        gp_samples = self.gp_samples
        samples = Predictive(self.model, num_samples=gp_samples)(rng_key, X_new)
        n = samples["y"].shape[-1]
        samples["y"] = samples["y"].reshape(-1, 1, n)
        return samples

    def _sample_conditioned_prior(self, rng_key, X_new):
        f = self.kernel.sample_parameters
        samples = Predictive(f, num_samples=self.hp_samples)(rng_key)
        predictive = jax.vmap(
            lambda p: self._sample(p[0], X_new, p[1], self.gp_samples)
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

        if self.data is None:
            return self._sample_unconditioned_prior(rng_key, x_new)
        if not self.model_fit:
            return self._sample_conditioned_prior(rng_key, x_new)
        return self._sample_posterior(rng_key, x_new)

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

        samples = self.sample(rng_key, x_new)
        y = samples["y"]
        return y.mean(axis=[0, 1]), jnp.var(y, axis=[0, 1])


class ExactGP(GaussianProcess):
    def __init__(self, kernel, data, observation_noise=False, mcmc=None):
        super().__init__(kernel, data, observation_noise)
        self.mcmc = mcmc

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
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(rng_key, self.x, self.y, **mcmc_run_kwargs)
        if print_summary:
            self.mcmc.print_summary()
        self.model_fit = True

    def _sample_posterior(self, rng_key, x_new):
        samples = self.mcmc.get_samples(group_by_chain=False)
        chain_length = len(next(iter(samples.values())))
        predictive = jax.vmap(
            lambda p: self._sample(p[0], x_new, p[1], self.gp_samples)
        )
        keys = jrp.split(rng_key, chain_length)
        sampled = predictive((keys, samples))
        return {**samples, "y": sampled}


class VariationalInferenceGP(GaussianProcess):
    def __init__(
        self,
        kernel,
        data,
        guide_factory=AutoNormal,
        svi=None,
        optimizer_factory=partial(numpyro.optim.Adam, step_size=1e-3, b1=0.5),
        loss_factory=Trace_ELBO,
        kernel_params=None,
        observation_noise=False,
    ):
        super().__init__(kernel, data, observation_noise)
        self.guide_factory = guide_factory
        self.svi = svi
        self.optimizer_factory = optimizer_factory
        self.loss_factory = loss_factory
        self.kernel_params = kernel_params

    def _print_summary(self):
        params_map = self.svi.guide.median(self.kernel_params.params)
        print("\nInferred GP parameters")
        for k, vals in params_map.items():
            spaces = " " * (15 - len(k))
            print(k, spaces, jnp.around(vals, 4))

    def fit(self, rng_key, num_steps, progress_bar=True, print_summary=True):
        optim = self.optimizer_factory()
        guide = self.guide_factory(self.model)
        loss = self.loss_factory()
        self.svi = SVI(
            self.model,
            guide=guide,
            optim=optim,
            loss=loss,
        )
        self.kernel_params = self.svi.run(
            rng_key, num_steps, progress_bar=progress_bar, x=self.x, y=self.y
        )
        if print_summary:
            self._print_summary()
        self.model_fit = True

    def _sample_posterior(self, rng_key, x_new):
        kernel_params = self.svi.guide.median(self.kernel_params.params)
        samples = self._sample(rng_key, x_new, kernel_params)
        samples = samples.reshape(1, samples.shape[0], samples.shape[1])
        kernel_params["y"] = samples
        return kernel_params

    def predict(self, rng_key, x_new):
        # Override the default sampling behavior if the model is fit and
        # the data is provided. SVI is special in that there is only the
        # median kernel parameters to consider
        if self.model_fit:
            kernel_params = self.svi.guide.median(self.kernel_params.params)
            mean, cov = self.get_mvn(x_new, kernel_params)
            return mean, cov.diagonal()

        return super().sample(rng_key, x_new)
