"""
kernels.py
==========

Kernel functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
    Update starting in August 2024 to change all kernels to classes
    instead of functions.
"""

import math

import jax.numpy as jnp
from jax import jit
from monty.json import MSONable


def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)


def add_jitter(x, jitter=1e-6):
    return x + jitter


def squared_distance(X, Z):
    """Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions

    Parameters
    ----------
    X, Z : jnp.ndarray
        The arrays between which to compute the distance.

    Returns
    -------
    jnp.ndarray
    """

    X2 = (X**2).sum(1, keepdims=True)
    Z2 = (Z**2).sum(1, keepdims=True)
    XZ = jnp.matmul(X, Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


class Kernel(MSONable):
    def __init__(self, noise=None, jitter=1e-6):
        self.noise = noise
        self.jitter = jitter


@jit
def _rbf_kernel(X, Z, k_scale, k_length, noise, jitter):
    r2 = squared_distance(X / k_length, Z / k_length)
    k = k_scale * jnp.exp(-0.5 * r2)
    if X.shape == Z.shape:
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


class RBFKernel(Kernel):
    """Radial basis function kernel.

    Parameters
    ----------
    k_scale : jnp.ndarray or float
        The absolute scale of the kernel function.
    k_length : jnp.ndarray or float
        The lengthscale of the kernel function.
    """

    def __init__(self, k_scale=1.0, k_length=1.0):
        self.k_scale = k_scale
        self.k_length = k_length

    def __call__(self, X, Z):
        """
        Parameters
        ----------
        X, Z : jnp.ndarray
            A 2d vector with dimension (N, N_features)

        Returns
        -------
        jnp.ndarray
        """

        return _rbf_kernel(
            X, Z, self.k_scale, self.k_length, self.noise, self.jitter
        )


@jit
def _matern_kernel(X, Z, k_scale, k_length, noise, jitter):
    r2 = squared_distance(X / k_length, Z / k_length)
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = k_scale * (1 + sqrt5_r + (5 / 3) * r2) * jnp.exp(-sqrt5_r)
    if X.shape == Z.shape:
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


class MaternKernel(Kernel):
    """Matern basis function kernel.

    Parameters
    ----------
    k_scale : jnp.ndarray or float
        The absolute scale of the kernel function.
    k_length : jnp.ndarray or float
        The lengthscale of the kernel function.
    """

    def __init__(self, k_scale=1.0, k_length=1.0):
        self.k_scale = k_scale
        self.k_length = k_length

    def __call__(self, X, Z):
        """
        Parameters
        ----------
        X, Z : jnp.ndarray
            A 2d vector with dimension (N, N_features)

        Returns
        -------
        jnp.ndarray
        """

        return _matern_kernel(
            X, Z, self.k_scale, self.k_length, self.noise, self.jitter
        )


@jit
def _periodic_kernel(X, Z, k_scale, k_length, k_period, noise, jitter):
    d = X[:, None] - Z[None]
    scaled_sin = jnp.sin(math.pi * d / k_period) / k_length
    k = k_scale * jnp.exp(-2 * (scaled_sin**2).sum(-1))
    if X.shape == Z.shape:
        k += add_jitter(noise, jitter) * jnp.eye(X.shape[0])
    return k


class PeriodicKernel(Kernel):
    """Periodic kernel.

    Parameters
    ----------
    k_scale : jnp.ndarray or float
        The absolute scale of the kernel function.
    k_length : jnp.ndarray or float
        The lengthscale of the kernel function.
    k_period : jnp.ndarray or float
        The period of the kernel function.
    """

    def __init__(self, k_scale=1.0, k_length=1.0, k_period=1.0):
        self.k_scale = k_scale
        self.k_length = k_length
        self.k_period = k_period

    def __call__(self, X, Z):
        """
        Parameters
        ----------
        X, Z : jnp.ndarray
            A 2d vector with dimension (N, N_features)

        Returns
        -------
        jnp.ndarray
        """

        return _periodic_kernel(
            X, Z, self.k_scale, self.k_length, self.noise, self.jitter
        )
