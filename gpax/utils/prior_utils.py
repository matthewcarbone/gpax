"""
prior_utils.py
==============

Utilities for priors and probabilistic functions used in numpyro. Specifically
contains abstractions over mean and kernel functions in the form of the
``Prior`` object.

Created by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

from functools import cached_property

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from attrs import define, field
from attrs.validators import ge, instance_of
from monty.json import MSONable


def get_parameter_prior(name, distribution, plate=1):
    """A utility function for getting a numpyro prior from a provided
    distribution. Floats and ints are interpreted as deterministic values.

    Parameters
    ----------
    name : str
        The name of the variable
    distribution : numpyro.distributions.Distribution or float or int
        The prior distribution to sample from. This is interpreted as
        numpyro.deterministic if float or int.
    plate : int
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
    if plate == 1:
        return numpyro.sample(name, distribution)

    with numpyro.plate(f"{name}_plate", plate):
        return numpyro.sample(name, distribution)


def get_parameter_as_float(parameter):
    """Converts a provided parameter, potentially a numpyro distribution,
    into a 'reduced' representation. This will either be simply the provided
    parameter if constant (float or int), or the mean of a distribution if
    parameter is of type numpyro.distributions.Distribution.

    Parameters
    ----------
    parameter : numpyro.distribution, int, float
        The parameter to retrieve in float representation.

    Returns
    -------
    float
        The mean of the input parameter distribution (or simply the value if
        the input is a constant float or int).
    """

    if isinstance(parameter, numpyro.distributions.Distribution):
        return parameter.mean
    return parameter


@define
class Parameter(MSONable):
    """Object containing a single parameter for a probabilistic model.
    Specifically, contains both the distribution (or constant) itself, as well
    as a value for ``plate``, which represents the number of independent
    instances of a variable to use during inference/training.

    Parameters
    ----------
    value : numpyro.distribution, int, float
        The value of the parameter itself, either a distribution or float/int.
    plate : int
        The number of independent instances of the variable. Used with
        numpyro.plate.
    """

    value = field(validator=instance_of((dist.Distribution, int, float)))
    plate = field(default=1, validator=[instance_of(int), ge(1)])

    def _get_parameter_prior(self, name):
        return get_parameter_prior(name, self.value, self.plate)


@define
class Prior(MSONable):
    """Base class representing any prior used in GPax. This can be, for
    instance, a mean prior, or a kernel prior, used in a Gaussian Process."""

    @cached_property
    def _params(self):
        keys = self.as_dict().keys()
        params = {}
        for key in keys:
            if "@" in key:
                continue
            obj = getattr(self, key)
            if isinstance(obj, Parameter):
                params[key] = obj
        return params

    @cached_property
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
            k: get_parameter_as_float(v.value) for k, v in self._params.items()
        }
