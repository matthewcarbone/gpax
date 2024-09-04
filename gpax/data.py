import json
from abc import ABC, abstractmethod
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from attrs import define, field
from attrs.validators import instance_of

from gpax.monty.json import MontyDecoder, MontyEncoder, MSONable

EPSILON = 1e-8
DATA_TYPES = (jnp.ndarray, np.ndarray, type(None))


def ensure_2d(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


@define
class ArrayMetadata(MSONable):
    transforms_as = field(default="mean", validator=instance_of(str))
    is_transformed = field(default=False, validator=instance_of(bool))


class Array(MSONable, np.ndarray):
    def __new__(cls, input_array):
        if hasattr(input_array, "metadata"):
            metadata = deepcopy(input_array.metadata)
        else:
            metadata = deepcopy(ArrayMetadata())
        obj = np.asarray(input_array).view(cls)  # this goes to finalize
        obj.metadata = deepcopy(metadata)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, "metadata", None)

    def as_dict(self):
        d = super().as_dict()
        arr = np.array(self)
        return {
            **d,
            "!array_data": arr,
            "!metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        data = MontyDecoder().process_decoded(d.pop("!array_data"))
        klass = cls(data)
        metadata = d.pop("!metadata")
        klass.metadata = deepcopy(ArrayMetadata.from_dict(metadata))
        return klass

    def to_json(self):
        d = self.as_dict()
        return json.dumps(d, cls=MontyEncoder)


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

    def _preprocess(self, x, fitting=False):
        """Preprocesses the input to a forward or reverse transformation.

        Parameters
        ----------
        x : array_like
            The input array to transform.
        fitting : bool
            Whether or not the function calling this one is fitting the
            transform or performing a transform.

        Returns
        -------
        array_like, None
        """

        if not self._is_fit and not fitting:
            raise RuntimeError("Transform must be fit before forward/reverse")
        if self._is_fit and fitting:
            raise RuntimeError("Cannot fit the transform twice")
        if x is None:
            return None
        if not hasattr(x, "metadata"):
            raise ValueError("Input array must be of type Array")
        x = ensure_2d(x)
        return x

    def forward(self, x):
        x = self._preprocess(x, False)
        if x is None:
            return None
        if x.metadata.is_transformed:
            return x
        t_as = x.metadata.transforms_as
        if t_as == "mean":
            new_x = self._forward_mean(x)
        elif t_as == "std":
            new_x = self._forward_std(x)
        else:
            raise ValueError(f"Array has invalid transform type {t_as}")
        new_x.metadata.is_transformed = True
        return new_x

    def reverse(self, x):
        x = self._preprocess(x, False)
        if x is None:
            return None
        if not x.metadata.is_transformed:
            return x
        t_as = x.metadata.transforms_as
        if t_as == "mean":
            new_x = self._reverse_mean(x)
        elif t_as == "std":
            new_x = self._reverse_std(x)
        else:
            raise ValueError(f"Array has invalid transform type {t_as}")
        new_x.metadata.is_transformed = False
        return new_x


@define
class IdentityTransform(Transform):
    def fit(self, x):
        self._preprocess(x, True)
        self._is_fit = True

    def _forward_mean(self, x):
        return x

    def _forward_std(self, x):
        return x

    def _reverse_mean(self, x):
        return x

    def _reverse_std(self, x):
        return x


@define
class ScaleTransform(Transform):
    """Simple transform class for scaling data, provided in the shape of
    N x n_dims, to minimum -1 and maximum 1 along each dimension."""

    minima = field(default=None, validator=instance_of(DATA_TYPES))
    maxima = field(default=None, validator=instance_of(DATA_TYPES))

    def fit(self, x):
        x = self._preprocess(x, True)
        if x is None:
            return
        self._is_fit = True
        self.minima = x.min(axis=0, keepdims=True)
        self.maxima = x.max(axis=0, keepdims=True)

    def _forward_mean(self, x):
        numerator = x - self.minima
        delta = self.maxima - self.minima + EPSILON
        return numerator / delta

    def _forward_std(self, x):
        delta = self.maxima - self.minima + EPSILON
        return x / delta

    def _reverse_mean(self, x):
        delta = self.maxima - self.minima + EPSILON
        return x * delta + self.minima

    def _reverse_std(self, x):
        delta = self.maxima - self.minima + EPSILON
        return x * delta


@define
class NormalizeTransform(Transform):
    mean = field(default=None, validator=instance_of(DATA_TYPES))
    std = field(default=None, validator=instance_of(DATA_TYPES))

    def fit(self, x):
        x = self._preprocess(x, True)
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
