# Installation

To install the latest, stable deployment of GPax, we recommend simply using the [PyPI distribution](https://pypi.org/project/gpax/) via `pip`:

```bash
pip install gpax
```

:::{note}
If you would like to utilize a GPU acceleration, follow these [instructions](https://github.com/google/jax#installation) to install JAX with a GPU support.
:::

:::{important}
This is the version 1 release of GPax, and included an overhaul of the entire API. To install the previous major version of GPax, use `pip install gpax<1`
:::

Otherwise, if you are confident in what's on the [main branch](https://github.com/ziatdinovmax/gpax) of GPax, you can also install directly from the GitHub repository:

```bash
pip install git+https://github.com/ziatdinovmax/gpax
```

However, we do not recommend this! Installing a specific version from PyPI is always a safer option.

:::{tip}
If you are a Windows user, we recommend to use the Windows Subsystem for Linux (WSL2), which comes free on Windows 10 and 11.
:::

