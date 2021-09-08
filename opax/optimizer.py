from typing import Any, Callable, Sequence, Type

from .schedule import LRSchedule, ScheduleOrFloat, lr_schedule
from .transform import (
    GradientTransformation,
    add_decayed_weights,
    chain,
    scale,
    scale_by_adam,
    scale_by_schedule,
    trace,
)


def _scale_lr(lr: ScheduleOrFloat):
    if isinstance(lr, LRSchedule):
        return scale_by_schedule(lr)
    else:
        return scale(lr)


def sgd(learning_rate: ScheduleOrFloat = 1e-2, momentum: float = 0.9):
    return chain(
        trace(momentum),
        _scale_lr(learning_rate),
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
        _scale_lr(learning_rate),
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
        _scale_lr(learning_rate),
    )
