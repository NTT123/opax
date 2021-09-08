from typing import Optional

from .schedule import ScheduleOrFloat
from .transform import (
    add_decayed_weights,
    chain,
    identity,
    scale,
    scale_by_adam,
    scale_by_rms,
    scale_by_schedule,
    scale_by_stddev,
    trace,
)


def _scale_by_learning_rate(lr: ScheduleOrFloat):
    if callable(lr):
        return scale_by_schedule(lr)
    else:
        return scale(lr)


def sgd(learning_rate: ScheduleOrFloat = 1e-2, momentum: float = 0.9):
    return chain(
        trace(momentum),
        _scale_by_learning_rate(learning_rate),
    )


def rmsprop(
    learning_rate: ScheduleOrFloat = 1e-4,
    decay_rate: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    centered: bool = False,
    momentum: Optional[float] = None,
):
    if centered:
        return chain(
            scale_by_stddev(
                decay_rate=decay_rate, eps=eps, initial_scale=initial_scale
            ),
            _scale_by_learning_rate(learning_rate),
            trace(decay_rate=momentum) if momentum is not None else identity(),
        )
    else:
        return chain(
            scale_by_rms(decay_rate=decay_rate, eps=eps, initial_scale=initial_scale),
            _scale_by_learning_rate(learning_rate),
            trace(decay_rate=momentum) if momentum is not None else identity(),
        )


def adam(
    learning_rate: ScheduleOrFloat = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
):
    return chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        _scale_by_learning_rate(learning_rate),
    )


def adamw(
    learning_rate: ScheduleOrFloat = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-4,
):
    return chain(
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        add_decayed_weights(weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    )
