"""
kernels.py
==========

Kernel functions

Note that the defined priors and parameters of the Kernel functions pertain
to data _after_ transforms specified in the GP are applied. In other words,
a k_length parameter of 1.0 would be the prior after the data is scaled or
transformed by whatever transformation is specified in the GP.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from attrs import define, field
from attrs.validators import gt, instance_of
from jax import jit
from monty.json import MSONable


def _add_jitter(x, jitter=1e-6):
    return x + jitter


def _squared_distance(X, Z, lengthscale):
    """Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions

    Parameters
    ----------
    X, Z : jnp.ndarray
        The arrays between which to compute the distance.

    Returns
    -------
    jnp.ndarray
    """

    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = scaled_X @ scaled_Z.T
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


def get_parameter_prior(name, distribution, plate_dims=1):
    """A utility function for getting a numpyro prior from a provided
    distribution. Floats and ints are interpreted as deterministic values.

    Parameters
    ----------
    name : str
        The name of the variable
    distribution : numpyro.distributions.Distribution or float or int
        The prior distribution to sample from. This is interpreted as
        numpyro.deterministic if float or int.
    plate_dims : int
        The number of dimensions to run the numpyro plate primitive over.
        Defaults to 1.

    Returns
    -------
    Some numpyro object depending on whether or not distribution is a true
    distribution or just some constant value.
    """

    if isinstance(distribution, (float, int)):
        return numpyro.deterministic(name, jnp.array(distribution))

    if not isinstance(distribution, numpyro.distributions.Distribution):
        raise ValueError(
            f"Provided distribution {distribution} was not of type float, "
            "int, or numpyro.distribution.Distribution, but is required to "
            "be one of these types."
        )

    # Otherwise, the provided distribution is a numpyro distribution object
    if plate_dims == 1:
        return numpyro.sample(name, distribution)

    with numpyro.plate(f"{name}_plate", plate_dims):
        return numpyro.sample(name, distribution)


def get_parameter_as_float(parameter):
    """Converts a provided parameter, potentially a numpyro distribution,
    into a 'reduced' representation. This will either be simply the provided
    parameter if constant (float or int), or the mean of a distribution if
    parameter is of type numpyro.distributions.Distribution."""

    if isinstance(parameter, numpyro.distributions.Distribution):
        return parameter.mean
    return parameter


@define
class Kernel(MSONable, ABC):
    k_noise = field(
        default=dist.LogNormal(0.0, 1.0),
        validator=instance_of((dist.Distribution, int, float)),
    )
    k_jitter = field(default=1e-6, validator=[instance_of(float), gt(0.0)])
    jit = field(default=True)

    @contextmanager
    def no_jit(self):
        self.jit = False
        try:
            yield
        finally:
            self.jit = True

    @abstractmethod
    def _kernel(self): ...

    @property
    def kernel(self):
        def _f(*args, **kwargs):
            return self._kernel(*args, **kwargs, jit=self.jit)

        if self.jit:
            return jit(_f)
        else:
            return _f


@define
class ScaleKernel(Kernel):
    k_scale = field(
        default=dist.LogNormal(0.0, 1.0),
        validator=instance_of((dist.Distribution, int, float)),
    )
    k_scale_dims = field(default=1, validator=[instance_of(int), gt(0)])
    k_length = field(
        default=dist.LogNormal(0.0, 1.0),
        validator=instance_of((dist.Distribution, int, float)),
    )
    k_length_dims = field(default=1, validator=[instance_of(int), gt(0)])

    @property
    def kernel_params(self):
        return {
            "k_scale": self.k_scale,
            "k_length": self.k_length,
            "k_noise": self.k_noise,
            "k_jitter": self.k_jitter,
        }

    def sample_parameters(self):
        return {
            "k_scale": get_parameter_prior(
                "k_scale", self.k_scale, self.k_scale_dims
            ),
            "k_length": get_parameter_prior(
                "k_length", self.k_length, self.k_length_dims
            ),
            "k_noise": get_parameter_prior("k_noise", self.k_noise, 1),
            "k_jitter": get_parameter_prior("k_jitter", self.k_jitter, 1),
        }

    def get_sample_parameter_means(self):
        return {
            "k_scale": get_parameter_as_float(self.k_scale),
            "k_length": get_parameter_as_float(self.k_length),
            "k_noise": get_parameter_as_float(self.k_noise),
            "k_jitter": get_parameter_as_float(self.k_jitter),
        }

    def sample_prior(self, X1, X2):
        """Radial basis function kernel prior. The parameters of this function
        are assumed to be distributions."""

        return self.kernel(X1, X2, **self.sample_parameters())


@define
class RBFKernel(ScaleKernel):
    @staticmethod
    def _kernel(
        X1,
        X2,
        k_scale,
        k_length,
        k_noise=0.0,
        k_jitter=1e-6,
        apply_noise=True,
        jit=True,
    ):
        if jit:
            _exp = jnp.exp
            _eye = jnp.eye
        else:
            _exp = np.exp
            _eye = np.eye
        r2 = _squared_distance(X1, X2, k_length)
        k = k_scale * _exp(-0.5 * r2)
        if X1.shape == X2.shape and apply_noise:
            k += _add_jitter(k_noise, k_jitter) * _eye(X1.shape[0])
        return k


@define
class MaternKernel(ScaleKernel):
    @staticmethod
    def _kernel(
        X,
        Z,
        k_scale,
        k_length,
        k_noise=0.0,
        k_jitter=1e-6,
        apply_noise=True,
        jit=True,
    ):
        if jit:
            _exp = jnp.exp
            _sqrt = jnp.sqrt
            _eye = jnp.eye
        else:
            _exp = np.exp
            _sqrt = np.sqrt
            _eye = np.eye
        r2 = _squared_distance(X, Z, k_length)
        r = _sqrt(r2 + 1e-12)
        sqrt5_r = 5**0.5 * r
        k = k_scale * (1 + sqrt5_r + (5 / 3) * r2) * _exp(-sqrt5_r)
        if X.shape == Z.shape and apply_noise:
            k += _add_jitter(k_noise, k_jitter) * _eye(X.shape[0])
        return k


# @jit
# def _matern_kernel(X, Z, k_scale, k_length, noise, jitter):
#     r2 = _squared_distance(X / k_length, Z / k_length)
#     r = _sqrt(r2)
#     sqrt5_r = 5**0.5 * r
#     k = k_scale * (1 + sqrt5_r + (5 / 3) * r2) * jnp.exp(-sqrt5_r)
#     if X.shape == Z.shape:
#         k += _add_jitter(noise, jitter) * jnp.eye(X.shape[0])
#     return k
#
#
# @jit
# def _periodic_kernel(X, Z, k_scale, k_length, k_period, noise, jitter):
#     d = X[:, None] - Z[None]
#     scaled_sin = jnp.sin(math.pi * d / k_period) / k_length
#     k = k_scale * jnp.exp(-2 * (scaled_sin**2).sum(-1))
#     if X.shape == Z.shape:
#         k += _add_jitter(noise, jitter) * jnp.eye(X.shape[0])
#     return k
