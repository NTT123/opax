"""PAX optimizer library."""

from . import schedule
from .optimizer import adam, adamw, rmsprop, sgd
from .transform import (
    GradientTransformation,
    add_decayed_weights,
    chain,
    clip,
    clip_by_global_norm,
    differentially_private_aggregate,
    scale,
    scale_by_adam,
    scale_by_rms,
    scale_by_schedule,
    scale_by_stddev,
    trace,
)
from .utils import apply_gradients, apply_updates, transform_gradients

__version__ = "0.2.11"

__all__ = (
    "adam",
    "adamw",
    "add_decayed_weights",
    "apply_gradients",
    "apply_updates",
    "chain",
    "clip_by_global_norm",
    "clip",
    "differentially_private_aggregate",
    "GradientTransformation",
    "rmsprop",
    "scale_by_adam",
    "scale_by_rms",
    "scale_by_schedule",
    "scale_by_stddev",
    "scale",
    "schedule",
    "sgd",
    "trace",
    "transform_gradients",
)
