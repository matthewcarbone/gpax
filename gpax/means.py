"""
Mean Functions
==============

Python classes that capture prior mean information. Users can define their
own mean functions by inheriting the base Mean class, and implementing the
_mean_function method, which takes as input a coordinate ``x`` and keyword
arguments ``**params``.

.. note::

    In order to register as a parameter that numpyro can learn during
    inference, it must begin with the prefix ``"m_"``. If it does not start
    with this prefix, it is assumed to be a constant to the class, e.g., the
    ``MeanPrior`` has a mean that is set to a constant at instantiation.

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from attrs import define, field
from attrs.validators import instance_of

from gpax.utils.prior_utils import MeanPrior


@define
class Mean(MeanPrior, ABC):
    @abstractmethod
    def _mean_function(self, x, **params):
        raise NotImplementedError

    def sample_prior(self, x):
        """Radial basis function kernel prior. The parameters of this function
        are assumed to be distributions."""

        return self.__call__(x, **self.sample_parameters())

    def __call__(self, x, **params):
        # The mean function should always be 1d
        return self._mean_function(x, **params).squeeze()


@define
class ConstantMean(Mean):
    mean = field(default=0.0, validator=instance_of((float, int)))

    def _mean_function(self, x):
        return jnp.ones(x.shape[0]) * self.mean
