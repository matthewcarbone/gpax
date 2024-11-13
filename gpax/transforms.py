"""
Various data transformations used in the GPax code to transform input and
output data in the GPs.

This module hosts a number of useful transforms which can be used "out of the
box". In addition, defining your own transform is easy! Simply inherit
{py:class}`Transform` and define the following five methods with the following
signatures:
```python
_fit(x: ArrayLike) -> None
_forward_mean(x: ArrayLike) -> ArrayLike
_reverse_mean(x: ArrayLike) -> ArrayLike
_forward_std(x: ArrayLike) -> ArrayLike
_reverse_std(x: ArrayLike) -> ArrayLike
```

This is best demonstrated by example. Consider the
{py:obj}`NormalizeTransform` in this module. It has the above five methods as:

```python
def _fit(self, x):
    self._mean = x.mean(axis=0, keepdims=True)
    self._std = x.std(axis=0, keepdims=True)
    self._is_fit = True


def _forward_mean(self, x):
    denominator = self._std + _EPSILON
    return (x - self._mean) / denominator


def _forward_std(self, x):
    return x / self._std


def _reverse_mean(self, x):
    denominator = self._std + _EPSILON
    return x * denominator + self._mean


def _reverse_std(self, x):
    return x * self._std
```
"""

# Created by Matthew R. Carbone (email: x94carbone@gmail.com)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import jax.numpy as jnp
import numpy as np
from monty.json import MSONable
from numpy.typing import ArrayLike

_EPSILON = 1e-8


def _ensure_2d(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


@dataclass
class Transform(ABC, MSONable):
    """A base transform class upon which all other transforms are built.
    Implements the {py:obj}`forward` and {py:obj}`reverse` methods, which act
    on {py:obj}`ArrayLike` data."""

    _is_fit: bool = False

    @abstractmethod
    def _fit(self):
        raise NotImplementedError

    def fit(self, x: ArrayLike) -> None:
        """Fits the transform on the provided data, `x`. Depending on the
        type of transform, this might do nothing (as in the case of the
        {py:class}`IdentityTransform`), or it might set the minimum and
        maximum of the data along each dimension (as in the case of the
        {py:class}`ScaleTransform`).

        :::{warning}
        You can only call {py:obj}`fit` once on a single transform, otherwise
        a {py:class}`RuntimeError` will be raised.
        :::

        :param x:
            The input data to fit on. Must be `array_like`. Data should be
            of shape `(N, d)`, where `N` is the number of examples, and `d`
            is the dimensionality of each example.

        :raises RuntimeError: If {py:obj}`fit` is called more than once on
        a single transform.
        """

        if self._is_fit:
            raise RuntimeError("Cannot call fit on a Transform more than once")
        if x is None:
            return
        self._fit(x)

    @abstractmethod
    def _forward_mean(self, x: ArrayLike) -> ArrayLike:
        """Test docstring"""
        raise NotImplementedError

    @abstractmethod
    def _forward_std(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def _reverse_mean(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def _reverse_std(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def forward(
        self,
        x: ArrayLike | None,
        transforms_as: Literal["mean", "std"] = "mean",
    ) -> ArrayLike | None:
        """Executes a forward transform on the provided data

        Performs a forward transformation on the provided data by doing
        the following: First, if the transform is not fit already, a
        RuntimeError is raised. Next, if `x` is `None`, `None` is returned.
        `x` is then ensured to be at least 2d, and then, depending on
        whether or not `transforms_as` is `"mean"` or `"std"`, the
        corresponding transformation is executed. If `transforms_as` is set to

        an invalid value, a ValueError is raised.
        :param x:
            The input data to forward transform.
        :param transforms_as:
            Can be either "mean" or "std".

        :returns: Transformed data.

        :raises ValueError: If the provided transform type is invalid (not
        `"mean"` or `"std"`).

        :raises RuntimeError: If the transform is not fit.
        """

        if not self._is_fit:
            raise RuntimeError("Cannot execute transform before fit is called")
        if x is None:
            return None
        x = _ensure_2d(x)
        if transforms_as == "mean":
            return self._forward_mean(x)
        elif transforms_as == "std":
            return self._forward_std(x)
        else:
            raise ValueError(f"arr has invalid transform type {transforms_as}")

    def reverse(
        self,
        x: ArrayLike | None,
        transforms_as: Literal["mean", "std"] = "mean",
    ) -> ArrayLike | None:
        """Executes a reverse/inverse transform on the provided data

        Performs a reverse transformation on the provided data by performing
        the same series of steps as {py:class}`forward` above.

        :param x:
            The input data to reverse transform.
        :param transforms_as:
            Can be either "mean" or "std".

        :returns: Transformed data.

        :raises ValueError: If the provided transform type is invalid (not
        `"mean"` or `"std"`).

        :raises RuntimeError: If the transform is not fit.
        """

        if not self._is_fit:
            raise RuntimeError("Cannot execute transform before fit is called")
        if x is None:
            return None
        x = _ensure_2d(x)
        if transforms_as == "mean":
            return self._reverse_mean(x)
        elif transforms_as == "std":
            return self._reverse_std(x)
        else:
            raise ValueError(f"arr has invalid transform type {transforms_as}")


@dataclass
class IdentityTransform(Transform):
    """The simplest transform: does nothing. However, it must still follow
    the same rules as the other transforms. I.e., {py:obj}`Transform.fit` must
    still be called before attempting to execute the transform."""

    def _fit(self, _):
        self._is_fit = True

    def _forward_mean(self, x):
        return x

    def _forward_std(self, x):
        return x

    def _reverse_mean(self, x):
        return x

    def _reverse_std(self, x):
        return x


_TMIN = -1.0
_TMAX = 1.0


@dataclass
class ScaleTransform(Transform):
    """Transformation class for scaling data, provided in the shape of shape
    `(N, d)`, to minimum -1 and maximum 1 along each dimension

    Given data $x \in \mathbb{R}^{N \\times d}$, with a vector of minima and
    maxima of $x_\\mathrm{min}$ and $x_\\mathrm{max}$ along each axis, the
    transformed data $x'$ will be given by the transformation,

    $$ x' = 2\\frac{x-x_\\mathrm{min}}{x_\\mathrm{max} -
    x_\\mathrm{min}} - 1. $$

    Similarly, the standard deviation/error in $x$ transforms like

    $$ \delta x' = \\frac{2}{x_\\mathrm{max} - x_\\mathrm{min}} \delta x.$$

    These are the forward transforms. The reverse transforms can be easily
    derived by inverting these equations.
    """

    _minima: Union[jnp.ndarray, np.ndarray, float, int, None] = None
    _maxima: Union[jnp.ndarray, np.ndarray, float, int, None] = None

    def _fit(self, x):
        self._minima = x.min(axis=0, keepdims=True)
        self._maxima = x.max(axis=0, keepdims=True)
        self._is_fit = True

    def _forward_mean(self, x):
        numerator = x - self._minima
        delta_r = self._maxima - self._minima + _EPSILON
        delta_t = _TMAX - _TMIN
        return numerator / delta_r * delta_t + _TMIN

    def _forward_std(self, x):
        delta_r = self._maxima - self._minima + _EPSILON
        delta_t = _TMAX - _TMIN
        return x / delta_r * delta_t

    def _reverse_mean(self, x):
        delta_r = self._maxima - self._minima + _EPSILON
        delta_t = _TMAX - _TMIN + _EPSILON
        num = x * delta_r + self._minima * delta_t - _TMIN * delta_r
        return num / delta_t

    def _reverse_std(self, x):
        delta_r = self._maxima - self._minima + _EPSILON
        delta_t = _TMAX - _TMIN + _EPSILON
        return x * delta_r / delta_t


@dataclass
class NormalizeTransform(Transform):
    """Transformation class for normalizing data to the standard normal

    Given data $x \in \mathbb{R}^{N \\times d}$, with mean and standard
    deviation along each axis $\mu$ and $\sigma$, respectively, the
    transformation to the standard normal is given by:

    $$ x' = \\frac{x - \mu}{\sigma + \\varepsilon},$$

    where $\\varepsilon$ is a small number to ensure numerical stability when
    $\sigma$ is small.

    Similarly, the standard deviation transforms like

    $$ \delta x' = \\frac{1}{\sigma + \\varepsilon}\delta x.$$

    These are the forward transforms. The reverse transforms can be easily
    derived by inverting these equations.
    """

    _mean: Union[jnp.ndarray, np.ndarray, float, int, None] = None
    _std: Union[jnp.ndarray, np.ndarray, float, int, None] = None

    def _fit(self, x):
        self._mean = x.mean(axis=0, keepdims=True)
        self._std = x.std(axis=0, keepdims=True)
        self._is_fit = True

    def _forward_mean(self, x):
        denominator = self._std + _EPSILON
        return (x - self._mean) / denominator

    def _forward_std(self, x):
        return x / self._std

    def _reverse_mean(self, x):
        denominator = self._std + _EPSILON
        return x * denominator + self._mean

    def _reverse_std(self, x):
        return x * self._std
