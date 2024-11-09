"""
means.py
==========

Python classes that capture prior mean information

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from attrs import define, field
from attrs.validators import instance_of

from gpax.utils.prior_utils import Parameter, Prior


@define
class Mean(Prior, ABC):
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
    mean = field(default=0.0, validator=[instance_of((float, int))])

    def _mean_function(self, x):
        return jnp.ones(x.shape[0]) * self.mean
