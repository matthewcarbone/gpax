"""
Kernels used for GPs.

:::{note}
The defined priors and parameters of the Kernel functions pertain to data
_after_ transforms specified in the GP are applied. In other words,
a k_length parameter of 1.0 would be the prior after the data is scaled or
transformed by whatever transformation is specified in the GP.
:::

:::{note}
In order to register as a parameter that numpyro can learn during
inference, it must begin with the prefix ``"k_"``. If it does not start
with this prefix, it is assumed to be a constant to the class.
:::
"""

# Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
# Modified by Matthew R. Carbone (email: x94carbone@gmail.com)

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro
from attrs import define, field
from attrs.validators import instance_of
from numpy.typing import ArrayLike
from numpyro import distributions as dist

from gpax.utils.prior_utils import Parameter, Prior


def _add_jitter(x: ArrayLike, jitter: float = 1e-6) -> ArrayLike:
    """Adds a constant jitter to the input, `x`."""

    return x + jitter


def _squared_distance(
    x1: ArrayLike, x2: ArrayLike, lengthscale: ArrayLike
) -> ArrayLike:
    """Computes a square of scaled distance, between `x1` and `x2`, which are
    vectors of shape `(N, d)` dimensions."""

    scaled_X = x1 / lengthscale
    scaled_Z = x2 / lengthscale
    X2 = (scaled_X**2).sum(1, keepdims=True)
    Z2 = (scaled_Z**2).sum(1, keepdims=True)
    XZ = scaled_X @ scaled_Z.T
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


@define
class Kernel(Prior, ABC):
    """Base kernel object. All kernels should inherit from this class.

    This base kernel defines two learnable parameters:
    * `k_noise` is the prior for the noise on the diagonal of the covariance
    matrix. By default, this is a Half Normal distribution, with `plate=1` and
    scale parameter 0.01.
    * `k_jitter` is the prior over the jitter on the diagonal of the
    covariance matrix. This is set to a constant of 1e-6 by default. It is
    strongly recommended to not change this parameter, as it's used purely
    for numerical stability when the diagonals of the covariance matrix are
    close to 0.
    """

    k_noise = field(
        default=Parameter(dist.HalfNormal(0.01), 1),
        validator=instance_of(Parameter),
    )
    k_jitter = field(
        default=Parameter(1.0e-6, 1), validator=instance_of(Parameter)
    )

    @property
    def attribute_prefix(self) -> str:
        """Returns `"k_"`, which sets the prefix for class attributes that
        are considered learnable during inference."""
        return "k_"

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
    """A kernel which defines the `k_scale` and `k_length` parameters."""

    k_scale: Parameter = field(
        default=Parameter(numpyro.distributions.LogNormal(0.0, 1.0), 1)
    )

    k_length: Parameter = field(
        default=Parameter(numpyro.distributions.LogNormal(0.0, 1.0), 1)
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


@define
class PeriodicKernel(ScaleKernel):
    def _kernel_function(
        self,
        X,
        Z,
        k_scale,
        k_length,
        k_period,
        k_noise=0.0,
        k_jitter=1e-6,
        apply_noise=True,
    ):
        d = X[:, None] - Z[None]
        scaled_sin = jnp.sin(jnp.pi * d / k_period) / k_length
        k = k_scale * jnp.exp(-2 * (scaled_sin**2).sum(-1))
        if X.shape == Z.shape and apply_noise:
            k += _add_jitter(k_noise, k_jitter) * jnp.eye(X.shape[0])
        return k
