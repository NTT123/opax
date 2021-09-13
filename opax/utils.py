import functools
from typing import TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


# Source: https://github.com/deepmind/jmp/blob/fdfcb830de8331f90de289edb18355964ec1f9f9/jmp/_src/loss_scale.py#L187
# This function is under the Apache License 2.0
def select_tree(pred: jnp.ndarray, a: T, b: T) -> T:
    """Selects a pytree based on the given predicate."""
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_multimap(functools.partial(jax.lax.select, pred), a, b)
