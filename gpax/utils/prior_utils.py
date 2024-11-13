"""Utilities for priors and probabilistic functions used in numpyro. Specifically
contains abstractions over mean and kernel functions in the form of the
{py:class}`Prior` object.
"""

# Created by Matthew R. Carbone (email: x94carbone@gmail.com)

from abc import ABC, abstractproperty
from functools import cache
from typing import Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from monty.json import MSONable


def get_parameter_prior(
    name: str,
    distribution: Union[dist.Distribution, float, int],
    plate: int = 1,
):
    """A utility function for getting a NumPyro prior from a provided
    distribution

    Provides the NumPyro prior via {py:class}`numpyro.sample` from the
    distribution. Also accounts for if `plate>1`, in which case `plate`
    independent samples are drawn for that parameter. `float` and `int` are
    interpreted as deterministic values.

    :::{warning}
    This function should only be used within a NumPyro model. Otherwise, an
    error will be thrown.
    :::

    :::{note}
    For more details on how {py:class}`numpyro.sample` works, see the
    [NumPyro documentation](https://num.pyro.ai/en/latest/primitives.html#sample).
    :::

    :param name:
        The name of the variable.
    :param distribution:
        The prior distribution to sample from. This is interpreted as
        numpyro.deterministic if float or int.
    :param plate:
        The number of dimensions to run the numpyro plate primitive over.
        Defaults to 1.

    :returns: Some numpyro object depending on whether or not distribution is
    a true distribution or just some constant value.
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
    if plate == 1:
        return numpyro.sample(name, distribution)

    with numpyro.plate(f"{name}_plate", plate):
        return numpyro.sample(name, distribution)


def distribution_as_float(parameter: dist.Distribution) -> float:
    """Converts a provided parameter, potentially a numpyro distribution,
    into a scalar representation

    Converts a parameter into a float. This will either be simply the provided
    parameter if constant (`float` or `int`), or the mean of a distribution if
    parameter is of type {py:obj}`numpyro.distributions.Distribution`.

    :param parameter:
        The parameter to retrieve in float representation.

    :returns: The mean of the input parameter distribution (or simply the
    value if the input is a constant float or int).
    """

    if isinstance(parameter, numpyro.distributions.Distribution):
        return parameter.mean
    return parameter


class Parameter(MSONable):
    """A parameter used in distribution

    Object containing a single parameter for a probabilistic model.
    Specifically, contains both the distribution (or constant) itself, as well
    as a value for ``plate``, which represents the number of independent
    instances of a variable to use during inference/training.

    :param value:
        The value of the parameter itself, either a distribution or float/int.
    :param plate:
        The number of independent instances of the variable. Used with
        numpyro.plate.
    """

    def __init__(
        self, value: Union[dist.Distribution, int, float], plate: int = 1
    ):
        self.value = value
        self.plate = plate

    def _get_parameter_prior(self, name):
        return get_parameter_prior(name, self.value, self.plate)


class Prior(ABC, MSONable):
    """A prior abstract base class

    Base class representing any prior used in GPax. This can be, for
    instance, a mean prior, or a kernel prior, used in a Gaussian Process.

    :::{warning}
    Variable naming for the {py:class}`Prior` is very important. Any variable
    which does not begin with the {py:obj}`attribute_prefix` string will
    not be considered during inference.

    For example, in the {py:obj}`gpax.kernels.Kernel` objects, this prefix
    is set to `"k_"`. Thus, parameters like `k_scale` are considered during
    inference, whereasa a parameter such as `scale` will not be considered.
    :::
    """

    @abstractproperty
    def attribute_prefix(self) -> str:
        """Class attributes that start with this prefix will be considered
        during GP training."""

        raise NotImplementedError

    def _params(self):
        keys = self.as_dict().keys()
        params = {}
        for key in keys:
            if "@" in key or not key.startswith(self.attribute_prefix):
                continue
            obj = getattr(self, key)
            if isinstance(obj, Parameter):
                params[key] = obj
        return params

    @cache
    @property
    def params(self):
        """Returns a dictionary, indexed by the parameter name, of all
        prior parameters available to the model. The parameters are extracted
        by first iterating through all class attributes, keeping only those of
        type ``Parameter``. The distributions corresponding to the parameters
        are then returned. For access to the Parameter objects themselves,
        use ``_prior``."""

        return {k: v.value for k, v in self._params.items()}

    def sample_parameters(self):
        """Method for sampling the parameters. Returns a ``numpyro.sample``
        object for each parameter. Intelligently detects when a parameter is
        constant and uses ``numpyro.deterministic`` accordingly. This method
        should only be used in numpyro logic, otherwise, it will likely cause
        errors."""

        return {k: v._get_parameter_prior(k) for k, v in self._params.items()}

    def get_sample_parameter_means(self):
        """Primarily a utility method for getting the means of numpyro
        distributions for use in variational inference. Returns the means of
        each distribution (or just the value itself if that distribution is
        a constant)."""

        return {
            k: distribution_as_float(v.value) for k, v in self._params.items()
        }


class MeanPrior(Prior):
    """Simple mean prior which sets the {py:obj}`attribute_prefix` to `"m_"`"""

    @property
    def attribute_prefix(self) -> str:
        return "m_"


class KernelPrior(Prior):
    """Simple mean prior which sets the {py:obj}`attribute_prefix` to `"k_"`"""

    @property
    def attribute_prefix(self) -> str:
        return "k_"
