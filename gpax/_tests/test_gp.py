import pytest

from gpax import state
from gpax.gp import ExactGP, VariationalInferenceGP
from gpax.kernels import RBFKernel


@pytest.mark.parametrize("gp_factory", [ExactGP, VariationalInferenceGP])
@pytest.mark.parametrize("kernel_factory", [RBFKernel])
@pytest.mark.parametrize("y_std", [None, 0.0, 1.0])
@pytest.mark.parametrize(
    "model_state", ["unconditioned_prior", "prior", "posterior"]
)
def test_ExactGP_sample_shapes(
    dummy_sinusoidal_data, gp_factory, kernel_factory, y_std, model_state
):
    HP_SAMPLES = 17
    GP_SAMPLES = 33
    KWARGS = {
        "hp_samples": HP_SAMPLES,
        "gp_samples": GP_SAMPLES,
    }
    if gp_factory == ExactGP:
        KWARGS["num_warmup"] = 100
    else:
        HP_SAMPLES = 1
        KWARGS["hp_samples"] = HP_SAMPLES
    x, y, x_grid = dummy_sinusoidal_data
    state.set_rng_key(0)
    kernel = kernel_factory()
    if model_state == "unconditioned_prior":
        if y_std is not None:
            with pytest.raises(ValueError):
                gp_factory(kernel=kernel, x=None, y=None, y_std=y_std, **KWARGS)
            return
        gp = gp_factory(kernel=kernel, x=None, y=None, y_std=y_std, **KWARGS)
    elif model_state == "prior":
        gp = gp_factory(kernel=kernel, x=x, y=y, y_std=y_std, **KWARGS)
    else:
        gp = gp_factory(kernel=kernel, x=x, y=y, y_std=y_std, **KWARGS)
        gp.fit()

    result = gp.sample(x_grid)
    y = result.y
    assert y.ndim == 3
    assert y.shape[0] == HP_SAMPLES
    assert y.shape[1] == GP_SAMPLES
