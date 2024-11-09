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

import jax.numpy as jnp
import numpyro.distributions as dist
from attrs import define, field
from attrs.validators import instance_of

from gpax.utils.prior_utils import Parameter, Prior


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


@define
class Kernel(Prior, ABC):
    """Base kernel object. All kernels should inherit from this class. Note
    that any class attribute beginning with ``k_`` will be interpreted as
    a kernel parameter (either a numpyro distribution or a constant)."""

    k_noise = field(
        default=Parameter(dist.HalfNormal(0.01), 1),
        validator=instance_of(Parameter),
    )
    k_jitter = field(
        default=Parameter(1.0e-6, 1), validator=instance_of(Parameter)
    )

    def sample_prior(self, x1, x2):
        """Radial basis function kernel prior. The parameters of this function
        are assumed to be distributions."""

        return self.__call__(x1, x2, **self.sample_parameters())

    @abstractmethod
    def _kernel_function(self, x1, x2, **params):
        raise NotImplementedError

    def __call__(self, x1, x2, **params):
        return self._kernel_function(x1, x2, **params)


@define
class ScaleKernel(Kernel):
    k_scale = field(
        default=Parameter(dist.LogNormal(0.0, 1.0)),
        validator=instance_of(Parameter),
    )
    k_length = field(
        default=Parameter(dist.LogNormal(0.0, 1.0)),
        validator=instance_of(Parameter),
    )


@define
class RBFKernel(ScaleKernel):
    def _kernel_function(
        self,
        X1,
        X2,
        k_scale,
        k_length,
        k_noise=0.0,
        k_jitter=1e-6,
        apply_noise=True,
    ):
        r2 = _squared_distance(X1, X2, k_length)
        k = k_scale * jnp.exp(-0.5 * r2)
        if X1.shape == X2.shape and apply_noise:
            k += _add_jitter(k_noise, k_jitter) * jnp.eye(X1.shape[0])
        return k


@define
class MaternKernel(ScaleKernel):
    def _kernel_function(
        self,
        X,
        Z,
        k_scale,
        k_length,
        k_noise=0.0,
        k_jitter=1e-6,
        apply_noise=True,
    ):
        r2 = _squared_distance(X, Z, k_length)
        r = jnp.sqrt(r2 + 1e-12)
        sqrt5_r = 5**0.5 * r
        k = k_scale * (1 + sqrt5_r + (5 / 3) * r2) * jnp.exp(-sqrt5_r)
        if X.shape == Z.shape and apply_noise:
            k += _add_jitter(k_noise, k_jitter) * jnp.eye(X.shape[0])
        return k


#
# @jit
# def _periodic_kernel(X, Z, k_scale, k_length, k_period, noise, jitter):
#     d = X[:, None] - Z[None]
#     scaled_sin = jnp.sin(math.pi * d / k_period) / k_length
#     k = k_scale * jnp.exp(-2 * (scaled_sin**2).sum(-1))
#     if X.shape == Z.shape:
#         k += _add_jitter(noise, jitter) * jnp.eye(X.shape[0])
#     return k
