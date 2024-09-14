import numpy as np
import pytest

from gpax.transforms import (
    IdentityTransform,
    NormalizeTransform,
    ScaleTransform,
)


def _assert_transform(transform, transforms_as, x):
    transform.fit(x)
    x1 = transform.forward(x, transforms_as=transforms_as)
    x2 = transform.reverse(x1, transforms_as=transforms_as)
    assert np.allclose(x, x2)
    return x1, x2


TRANSFORMS = [ScaleTransform, NormalizeTransform, IdentityTransform]


@pytest.mark.parametrize("transforms_as", ["mean", "std"])
@pytest.mark.parametrize("transform_factory", TRANSFORMS)
def test_other_transforms(dummy_data, transform_factory, transforms_as):
    x, _ = dummy_data
    transform = transform_factory()
    x1, _ = _assert_transform(transform, transforms_as, x)
    if transforms_as == "mean" and transform_factory == ScaleTransform:
        x1_min = x1.min(axis=0)
        x1_max = x1.max(axis=0)
        tmin_arr = -np.ones(shape=x1_min.shape)
        tmax_arr = np.ones(shape=x1_max.shape)
        assert np.allclose(tmin_arr, x1_min)
        assert np.allclose(tmax_arr, x1_max)
