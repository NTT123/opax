import math
from typing import Callable, Optional, Union

import jax.numpy as jnp
import pax

ScheduleFn = Callable[[jnp.ndarray], jnp.ndarray]


class LRSchedule(pax.Module):
    count: jnp.ndarray
    learning_rate: jnp.ndarray

    def __init__(self, schedule_fn):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.register_state("count", jnp.array(0, dtype=jnp.int32))
        self.register_state("learning_rate", jnp.array(0.0, dtype=jnp.float32))

    def step(self):
        self.count = self.count + 1
        self.learning_rate = self.schedule_fn(self.count)
        return self.learning_rate


def lr_schedule(schedule_fn: ScheduleFn):
    return LRSchedule(schedule_fn)


ScheduleOrFloat = Union[LRSchedule, float]


def exponential_decay(
    initial_value: float,
    step_size: float,
    reduce_factor: float = 2.0,
    min_value: Optional[float] = None,
):
    """Exponential decay of lr.

    lr will be reduced by a factor of `reduce_factor` for every `step_size` steps.

    Arguments:
        initial_value: The initial value of learning rate.
        step_size: number of steps to reach a `reduce_factor` scaling.
        reduce_factor: the scaling factor per `step_size`.
        min_value: the minimum learning rate.
    """

    def _schedule_fn(step: jnp.ndarray):
        scale = 1.0 / step_size * math.log(reduce_factor)
        lr = jnp.exp(-step.astype(jnp.float32) * scale) * initial_value
        if min_value is not None:
            lr = jnp.clip(lr, a_min=min_value)
        return lr

    return LRSchedule(_schedule_fn)
