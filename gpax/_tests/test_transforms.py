import numpy as np

from gpax.transforms import ScaleTransform


def test_ScaleTransform(dummy_data):
    data, _ = dummy_data
    transform = ScaleTransform()
    transform.fit(data)
    data_prime = transform.forward(data)
    data_prime_prime = transform.reverse(data_prime)
    assert np.allclose(data, data_prime_prime)
