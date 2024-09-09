import numpy as np
import pytest

from gpax.transforms import (
    IdentityTransform,
    NormalizeTransform,
    ScaleTransform,
)

MINIMA = [-123.0, -1.0, 0.5534]
MAXIMA = [100.0, 1.0, 0.8989]


def _assert_transform(transform, transforms_as, x):
    transform.fit(x)
    x1 = transform.forward(x, transforms_as=transforms_as)
    x2 = transform.reverse(x1, transforms_as=transforms_as)
    assert np.allclose(x, x2)
    return x1, x2


@pytest.mark.parametrize("transforms_as", ["mean", "std"])
@pytest.mark.parametrize("min_value", MINIMA)
@pytest.mark.parametrize("max_value", MAXIMA)
def test_ScaleTransform(dummy_data, transforms_as, min_value, max_value):
    x, _ = dummy_data
    transform = ScaleTransform(tmin=min_value, tmax=max_value)
    x1, _ = _assert_transform(transform, transforms_as, x)
    if transforms_as == "mean":
        x1_min = x1.min(axis=0)
        x1_max = x1.max(axis=0)
        tmin_arr = np.ones(shape=x1_min.shape) * min_value
        tmax_arr = np.ones(shape=x1_max.shape) * max_value
        assert np.allclose(tmin_arr, x1_min)
        assert np.allclose(tmax_arr, x1_max)


OTHER_TRANSFORMS = [NormalizeTransform, IdentityTransform]


@pytest.mark.parametrize("transforms_as", ["mean", "std"])
@pytest.mark.parametrize("transform_factory", OTHER_TRANSFORMS)
def test_other_transforms(dummy_data, transform_factory, transforms_as):
    x, _ = dummy_data
    transform = transform_factory()
    _assert_transform(transform, transforms_as, x)
