from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from attrs import define, field
from attrs.validators import instance_of
from monty.json import MSONable

EPSILON = 1e-8
DATA_TYPES = (jnp.ndarray, np.ndarray, type(None), float, int)


def ensure_2d(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


@define
class Transform(ABC, MSONable):
    _is_fit = field(default=False, validator=instance_of(bool))

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def _forward_mean(self): ...

    @abstractmethod
    def _forward_std(self): ...

    @abstractmethod
    def _reverse_mean(self): ...

    @abstractmethod
    def _reverse_std(self): ...

    def forward(self, x, transforms_as="mean"):
        assert self._is_fit
        if x is None:
            return None
        x = ensure_2d(x)
        if transforms_as == "mean":
            return self._forward_mean(x)
        elif transforms_as == "std":
            return self._forward_std(x)
        else:
            raise ValueError(f"arr has invalid transform type {transforms_as}")

    def reverse(self, x, transforms_as="mean"):
        assert self._is_fit
        if x is None:
            return None
        x = ensure_2d(x)
        if transforms_as == "mean":
            return self._reverse_mean(x)
        elif transforms_as == "std":
            return self._reverse_std(x)
        else:
            raise ValueError(f"arr has invalid transform type {transforms_as}")


@define
class IdentityTransform(Transform):
    def fit(self, _):
        assert not self._is_fit
        self._is_fit = True

    def _forward_mean(self, x):
        assert self._is_fit
        return x

    def _forward_std(self, x):
        assert self._is_fit
        return x

    def _reverse_mean(self, x):
        assert self._is_fit
        return x

    def _reverse_std(self, x):
        assert self._is_fit
        return x


@define
class ScaleTransform(Transform):
    """Simple transform class for scaling data, provided in the shape of
    N x n_dims, to minimum -1 and maximum 1 along each dimension."""

    minima = field(default=None, validator=instance_of(DATA_TYPES))
    maxima = field(default=None, validator=instance_of(DATA_TYPES))
    tmin = field(default=-1.0, validator=instance_of(DATA_TYPES))
    tmax = field(default=1.0, validator=instance_of(DATA_TYPES))

    def fit(self, x):
        assert not self._is_fit
        if x is None:
            return
        self._is_fit = True
        self.minima = x.min(axis=0, keepdims=True)
        self.maxima = x.max(axis=0, keepdims=True)

    def _forward_mean(self, x):
        numerator = x - self.minima
        delta_r = self.maxima - self.minima + EPSILON
        delta_t = self.tmax - self.tmin
        return numerator / delta_r * delta_t + self.tmin

    def _forward_std(self, x):
        delta_r = self.maxima - self.minima + EPSILON
        delta_t = self.tmax - self.tmin
        return x / delta_r * delta_t

    def _reverse_mean(self, x):
        delta_r = self.maxima - self.minima + EPSILON
        delta_t = self.tmax - self.tmin + EPSILON
        num = x * delta_r + self.minima * delta_t - self.tmin * delta_r
        return num / delta_t

    def _reverse_std(self, x):
        delta_r = self.maxima - self.minima + EPSILON
        delta_t = self.tmax - self.tmin + EPSILON
        return x * delta_r / delta_t


@define
class NormalizeTransform(Transform):
    mean = field(default=None, validator=instance_of(DATA_TYPES))
    std = field(default=None, validator=instance_of(DATA_TYPES))

    def fit(self, x):
        assert not self._is_fit
        if x is None:
            return
        self._is_fit = True
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)

    def _forward_mean(self, x):
        denominator = self.std + EPSILON
        return (x - self.mean) / denominator

    def _forward_std(self, x):
        return x / self.std

    def _reverse_mean(self, x):
        denominator = self.std + EPSILON
        return x * denominator + self.mean

    def _reverse_std(self, x):
        return x * self.std
