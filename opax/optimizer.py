from typing import Any, Callable, Sequence, Type

from .gradient_transform import (
    GradientTransformation,
    add_decayed_weights,
    chain,
    scale,
    scale_by_adam,
    trace,
)


def sgd(learning_rate: float = 1e-2, momentum: float = 0.9):
    return chain(
        trace(momentum),
        scale(learning_rate),
    )


def adam(
    learning_rate: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
):
    return chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        scale(learning_rate),
    )


def adamw(
    learning_rate: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-4,
):
    return chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        add_decayed_weights(weight_decay=weight_decay),
        scale(learning_rate),
    )
