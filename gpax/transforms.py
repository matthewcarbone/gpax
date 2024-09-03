from abc import ABC, abstractmethod

from monty.json import MSONable

EPSILON = 1e-8


class Transform(ABC, MSONable):
    def __init__(self, params=None, is_fit=False):
        self.params = params
        if self.params is None:
            self.params = {}
        self.is_fit = is_fit

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def forward(self): ...

    @abstractmethod
    def reverse(self): ...


class ScaleTransform(Transform):
    """Simple transform class for scaling data, provided in the shape of
    N x n_dims, to minimum -1 and maximum 1 along each dimension."""

    def fit(self, x):
        assert x.ndim == 2
        self.params["minima"] = x.min(axis=0, keepdims=True)
        self.params["maxima"] = x.max(axis=0, keepdims=True)
        self.is_fit = True

    def forward(self, x):
        assert self.is_fit
        minima = self.params["minima"]
        maxima = self.params["maxima"]
        return ((x - minima) / (maxima - minima + EPSILON)).copy()

    def reverse(self, x):
        assert self.is_fit
        # data' = (data - min) / (max - min)
        # data'(max - min) = data - min
        # data = data'(max - min) + min
        minima = self.params["minima"]
        maxima = self.params["maxima"]
        return (x * (maxima - minima + EPSILON) + minima).copy()
