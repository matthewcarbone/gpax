import numpy as np
import pytest

from gpax.transforms import (
    IdentityTransform,
    NormalizeTransform,
    ScaleTransform,
)

TRANSFORMS = [ScaleTransform, NormalizeTransform, IdentityTransform]


@pytest.mark.parametrize("transform_factory", TRANSFORMS)
@pytest.mark.parametrize("transforms_as", ["mean", "std"])
def test_transforms(dummy_data, transform_factory, transforms_as):
    x, _ = dummy_data
    transform = transform_factory()
    transform.fit(x)
    x1 = transform.forward(x, transforms_as=transforms_as)
    x2 = transform.reverse(x1, transforms_as=transforms_as)
    assert np.allclose(x, x2)
