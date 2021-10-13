"""Utility functions."""

from typing import Any, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jmp
from pax import Module, assert_structure_equal, module_and_value

from .transform import GradientTransformation

TreeDef = Any

T = TypeVar("T")
K = TypeVar("K", bound=Module)
O = TypeVar("O", bound=Module)


def transform_gradients(grads: T, optimizer: O, params: T) -> Tuple[T, O]:
    """Transform gradients to updates using an optimizer.

    Arguments:
        grads: The gradients.
        optimizer: The gradient transformation.
        params: The trainable parameters.

    Returns:
        A pair ``(updates, optimizer)``

        - **updates** : The transformed gradients.
        - **optimizer** : The *updated* optimizer.
    """

    assert isinstance(optimizer, GradientTransformation), "Expecting an OPAX optimizer."
    optimizer, updates = module_and_value(optimizer)(grads, params=params)
    return updates, optimizer


def apply_updates(params: T, updates: T) -> T:
    """Update the parameters with updates.

    Arguments:
        params: The trainable parameters.
        updates: The transformed gradients.
    """
    assert_structure_equal(updates, params)
    return jax.tree_map(lambda u, p: p - u, updates, params)


def apply_gradients(
    model: T, optimizer: K, grads: T, all_finite: Optional[jnp.ndarray] = None
) -> Tuple[T, K]:
    """Update model and optimizer with gradients `grads`.

    Arguments:
        model: the model which contains trainable parameters.
        optimizer: the gradient transformation.
        grads: the gradients w.r.t to trainable parameters of `model`.
        all_finite: True if gradients are finite. Default: `None`.

    Returns:
        A pair ``(new_model, new_optimizer)``

        - **new_model**: the updated model.
        - **new_optimizer**: the updated optimizer.
    """
    assert isinstance(model, Module), "Expecting a PAX Module."
    assert type(model) is type(grads)
    params = model.parameters()
    updates, new_optimizer = transform_gradients(
        grads.parameters(), optimizer, params=params
    )
    new_params = apply_updates(params, updates=updates)

    if all_finite is not None:
        new_params, new_optimizer = jmp.select_tree(
            all_finite, (new_params, new_optimizer), (params, optimizer)
        )

    new_model = model.update_parameters(new_params)
    return new_model, new_optimizer
