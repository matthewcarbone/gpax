from abc import ABC, abstractmethod

from attrs import define, field
from attrs.validators import instance_of
from monty.json import MSONable

EPSILON = 1e-8


@define
class Transform(ABC, MSONable):
    params = field(factory=dict, validator=instance_of(dict))
    _is_fit = field(default=False, validator=instance_of(bool))

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def forward(self): ...

    @abstractmethod
    def reverse(self): ...

    @abstractmethod
    def forward_std(self): ...

    @abstractmethod
    def reverse_std(self): ...


class ScaleTransform(Transform):
    """Simple transform class for scaling data, provided in the shape of
    N x n_dims, to minimum -1 and maximum 1 along each dimension."""

    def fit(self, x):
        assert x.ndim == 2
        self.params["minima"] = x.min(axis=0, keepdims=True)
        self.params["maxima"] = x.max(axis=0, keepdims=True)
        self._is_fit = True

    def forward(self, x):
        assert self._is_fit
        minima = self.params["minima"]
        maxima = self.params["maxima"]
        return ((x - minima) / (maxima - minima + EPSILON)).copy()

    def reverse(self, x):
        assert self._is_fit
        # data' = (data - min) / (max - min)
        # data'(max - min) = data - min
        # data = data'(max - min) + min
        minima = self.params["minima"]
        maxima = self.params["maxima"]
        return (x * (maxima - minima + EPSILON) + minima).copy()

    def forward_std(self, x_std):
        assert self._is_fit
        delta = self.params["maxima"] - self.params["minima"] + EPSILON
        return x_std / delta

    def reverse_std(self, x_std):
        assert self._is_fit
        delta = self.params["maxima"] - self.params["minima"] + EPSILON
        return x_std * delta


class NormalizeTransform(Transform):
    def fit(self, x):
        assert x.ndim == 2
        self.params["mean"] = x.mean(axis=0, keepdims=True)
        self.params["std"] = x.std(axis=0, keepdims=True)
        self._is_fit = True

    def forward(self, x):
        assert self._is_fit
        denominator = self.params["std"] + EPSILON
        return (x - self.params["mean"]) / denominator

    def reverse(self, x):
        assert self._is_fit
        denominator = self.params["std"] + EPSILON
        return x * denominator + self.params["mean"]

    def forward_std(self, x_std):
        assert self._is_fit
        return x_std / self.params["std"]

    def reverse_std(self, x_std):
        assert self._is_fit
        return x_std * self.params["std"]
