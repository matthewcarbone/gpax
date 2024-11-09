"""
means.py
==========

Python classes that capture prior mean information

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp

from gpax.utils.prior_utils import Parameter, Prior


class Mean(Prior, ABC):
    @staticmethod
    @abstractmethod
    def mean(x):
        raise NotImplementedError

    def sample_prior(self, X):
        """Radial basis function kernel prior. The parameters of this function
        are assumed to be distributions."""

        return self.mean(X, **self.sample_parameters())


class ZeroMean(Mean):
    @staticmethod
    def mean(x):
        return jnp.zeros(x.shape[0])
