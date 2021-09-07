"""Pax optimizer library."""

from typing import Sequence
from .gradient_transformers import scale, chain, clip, GradientTransformer


def chain_optimizer(*transformers: Sequence[GradientTransformer]):
    return lambda params: chain(*transformers)(params).build_optimizer()
