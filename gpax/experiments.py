"""
experiments.py
=======

Module containing an Experiment abstraction as well as a few "dummy" datasets
which can be used for testing and demonstrating GP-based Bayesian optimization
(or really any optimization).

Created by Matthew R. Carbone (email: x94carbone@gmail.com)

BSD 3-Clause License

Copyright (c) 2022, Brookhaven Science Associates, LLC, Brookhaven National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from attrs import define, field, frozen
from attrs.validators import ge, instance_of, optional
from monty.json import MSONable
from scipy.stats import qmc


@frozen
class ExperimentProperties(MSONable):
    """Defines the core set of experiment properties, which are frozen after
    setting. These are also serializable and cannot be changed after they
    are set."""

    n_input_dim = field(validator=instance_of(int))
    n_output_dim = 1
    domain = field(validator=optional(instance_of(np.ndarray)))


def scale_by_domain(x, domain):
    """Scales provided data x by the bounds provided in the domain. Note that
    all dimensions must perfectly match.

    Parameters
    ----------
    x : array_like
        The input data of shape (N, d). This data should be scaled between 0
        and 1.
    domain : array_like
        The domain to scale to. Should be of shape (2, d), where domain[0, :]
        is the minimum along each axis and domain[1, :] is the maximum.

    Returns
    -------
    np.ndarray
        The scaled data of shape (N, d).
    """

    if x.ndim != 2:
        raise ValueError("Dimension of x must be == 2")

    if domain.shape[1] != x.shape[1]:
        raise ValueError("Domain and x shapes mismatched")

    if domain.shape[0] != 2:
        raise ValueError("Domain shape not equal to 2")

    if x.min() < 0.0:
        raise ValueError("x.min() < 0 (should be >= 0)")

    if x.max() > 1.0:
        raise ValueError("x.max() > 0 (should be <= 0)")

    return (domain[1, :] - domain[0, :]) * x + domain[0, :]


def get_coordinates(points_per_dimension, domain):
    """Gets a grid of equally spaced points on each dimension.
    Returns these results in coordinate representation.

    Parameters
    ----------
    points_per_dimension : int or list
        The number of points per dimension. If int, assumed to be 1d.
    domain : np.ndarray
        A 2 x d array indicating the domain along each axis.

    Returns
    -------
    np.ndarray
        The available points for sampling.
    """

    if isinstance(points_per_dimension, int):
        points_per_dimension = [points_per_dimension] * domain.shape[1]
    gen = product(*[np.linspace(0.0, 1.0, nn) for nn in points_per_dimension])
    return scale_by_domain(np.array([xx for xx in gen]), domain)


def get_random_points(domain, n=1):
    """Gets a random selection of points on a provided domain. The dimension
    of the data is inferred from the shape of the domain.

    Parameters
    ----------
    domain : np.ndarray
        The domain to scale to. Should be of shape (2, d).
    n : int
        Total number of points.

    Returns
    -------
    np.ndarray
    """

    X = np.random.random(size=(n, domain.shape[1]))
    return scale_by_domain(X, domain)


def get_latin_hypercube_points(domain, n=5, seed=None):
    """Gets a random selection of points in the provided domain using the
    Latin Hypercube sampling algorithm.

    Parameters
    ----------
    domain : np.ndarray
        The domain to scale to. Should be of shape (2, d).
    n : int
        Total number of points.

    Returns
    -------
    np.ndarray
    """

    sampler = qmc.LatinHypercube(
        d=domain.shape[1], optimization="random-cd", seed=seed
    )
    sample = sampler.random(n=n)
    return qmc.scale(sample, *domain)


@define
class Experiment(ABC, MSONable):
    """Abstract base class for an experiment, which can be loosely thought of
    as a class which has a single useful method: __call__. A call to the
    class produces the result of an 'experiment' at the provide (N x d) points
    x. The output is always a (N,)-shape array containing the scalar
    observations at each point x."""

    metadata = field(factory=dict)
    properties = field(default=None)

    @abstractmethod
    def _truth(self, _):
        """Vectorized truth function. Should return the value of the truth
        function for all rows of the provided x input."""

        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def n_input_dim(self):
        return self.properties.n_input_dim

    @property
    def n_output_dim(self):
        return self.properties.n_output_dim

    @property
    def domain(self):
        return self.properties.domain

    def _validate_input(self, x):
        # Ensure x has the right shape
        if not x.ndim == 2:
            raise ValueError("x must have shape (N, d)")

        # Ensure that the second dimension of x is the right size for the
        # chosen experiment
        if not x.shape[1] == self.n_input_dim:
            raise ValueError(f"x second dimension must be {self.n_input_dim}")

        # Ensure that the input is in the bounds
        # Domain being None implies the domain is the entirety of the reals
        # This is a fast way to avoid the check
        if self.domain is not None:
            check1 = self.domain[0, :] <= x
            check2 = x <= self.domain[1, :]
            if not np.all(check1 & check2):
                raise ValueError("Some inputs x were not in the domain")

    def _validate_output(self, y):
        # Ensure y has the right shape
        if not y.ndim == 1:
            raise ValueError("y must have shape (N,)")

    def truth(self, x: np.ndarray) -> np.ndarray:
        """Access the noiseless results of an "experiment"."""

        self._validate_input(x)
        y = self._truth(x)
        self._validate_output(y)
        return y

    def get_random_coordinates(self, n=1):
        """Runs n random input points."""

        return get_random_points(self.properties.domain, n=n)

    def get_latin_hypercube_coordinates(self, n=5):
        """Gets n Latin Hypercube-random points."""

        return get_latin_hypercube_points(self.properties.domain, n)

    def get_dense_coordinates(self, ppd):
        """Gets a set of dense coordinates.

        Parameters
        ----------
        ppd : int or list
            Points per dimension.

        Returns
        -------
        np.ndarray
        """

        return get_coordinates(ppd, self.properties.domain)

    def get_domain_mpl_extent(self):
        """This is a helper for getting the "extent" for matplotlib's
        imshow.
        """

        if self.n_input_dim != 2:
            raise NotImplementedError("Only implemented for 2d inputs.")

        x0 = self.domain[0, 0]
        x1 = self.domain[1, 0]

        y0 = self.domain[0, 1]
        y1 = self.domain[1, 1]

        return [x0, x1, y0, y1]

    def __call__(self, x):
        """The result of the experiment."""

        return self.truth(x)


@define
class SimpleSinusoidal1d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1, domain=np.array([-1.1, 1.1]).reshape(-1, 1)
        )
    )
    noise = field(default=0.1, validator=[instance_of((float, int)), ge(0.0)])

    # FIX: this is a problem! You forgot the noise!!!
    def _truth(self, x):
        return np.atleast_1d(np.sin(10.0 * x).squeeze())

    def default_dataset(self, seed=0, num_init_points=25):
        np.random.seed(seed)

        x = np.random.uniform(-1.0, 1.0, num_init_points).reshape(-1, 1)
        # Generate noisy data from a known function
        f = self(x)
        y = f + np.random.normal(0.0, self.noise, num_init_points)
        y = y.squeeze()
        return x, y


@define
class SimpleDecayingSinusoidal1d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=1, domain=np.array([-3.0, 3.0]).reshape(-1, 1)
        )
    )

    def _truth(self, x):
        y = np.sin(5 * x + np.pi / 2) * np.exp(-(x**2))
        return np.atleast_1d(y.squeeze())


@define
class Simple2d(Experiment):
    properties = field(
        factory=lambda: ExperimentProperties(
            n_input_dim=2,
            domain=np.array([[-4.0, 5.0], [-5.0, 4.0]]).T,
        )
    )

    true_optima = np.array([2.0, -4.0])

    def _truth(self, x):
        x1 = x[:, 0]
        y1 = x[:, 1]
        res = (1 - x1 / 3.0 + x1**5 + y1**5) * np.exp(
            -(x1**2) - y1**2
        ) + 2.0 * np.exp(-((x1 - 2) ** 2) - (y1 + 4) ** 2)
        # Constant offset so that the minimum is roughly 0
        const = 0.737922
        y = (res + const) / (2.0 + const)
        return np.atleast_1d(y.squeeze())
