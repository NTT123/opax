import functools
from typing import TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")
U = TypeVar("U")


def apply_updates(params: T, updates: T) -> T:
    """Return the updated parameters."""
    return jax.tree_map(lambda p, u: p - u, params, updates)
