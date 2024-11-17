"""
GPax classes that capture prior mean information. Users can define their
own mean functions by inheriting the base Mean class, and implementing the
_mean_function method, which takes as input a coordinate `x` and keyword
arguments ``**params``.

:::{note}
In order to register as a parameter that numpyro can learn during
inference, it must begin with the prefix ``"m_"``. If it does not start
with this prefix, it is assumed to be a constant to the class, e.g., the
{py:class}`gpax.means.ConstantMean` has a mean that is set to a
constant at instantiation.
:::
"""

# Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
# Modified by Matthew R. Carbone (email: x94carbone@gmail.com)

from abc import ABC, abstractmethod

import jax.numpy as jnp
from attrs import define, field

from gpax.utils.prior_utils import Prior


@define
class Mean(Prior, ABC):
    """Base class for mean functions used in GPs.

    The {py:class}`Mean` class is a simple abstraction for representing
    probabilistic functions of one variable. At minimum, all classes
    inheriting this base must define the

    ```python
    _mean_function(self, x, **params)
    ```

    method. This mean function takes a single variable, `x`, of shape
    `(N, d)` as input, and returns the value of the mean function at each
    of the `N` points.
    """

    @property
    def attribute_prefix(self) -> str:
        """Returns `"m_"`, which sets the prefix for class attributes that
        are considered learnable during inference."""

        return "m_"

    @abstractmethod
    def _mean_function(self, x, **params):
        raise NotImplementedError

    def sample_prior(self, x):
        """Radial basis function mean prior. The parameters of this function
        are assumed to be distributions."""

        return self.__call__(x, **self.sample_parameters())

    def __call__(self, x, **params):
        # The mean function should always be 1d
        return self._mean_function(x, **params).squeeze()


@define
class ConstantMean(Mean):
    """The simplest mean function. Returns a constant of the input shape
    at every point.

    :param mean:
        The value of the mean function to return at every point.
    """

    mean: float | int = field(default=1.0)

    def _mean_function(self, x):
        return jnp.ones(x.shape[0]) * self.mean
