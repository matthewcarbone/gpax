"""
utils.py
========

Utility functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
Modified by Matthew R. Carbone (email: x94carbone@gmail.com)
"""

import jax


def enable_x64():
    """Use double (x64) precision for jax arrays"""
    jax.config.update("jax_enable_x64", True)


def split_array(x, s=100):
    """Splits an array along its 0th axis into parts [roughly] equal to the
    provided batch size, s."""

    return [x[i : i + s, ...] for i in range(0, len(x), s)]
