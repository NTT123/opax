"""Gradient Transformers."""

import jax
import pax
from typing import Any, Sequence, NamedTuple, Iterable
import jax.numpy as jnp

from .optimizer import GradientTransformOptimizer


class GradientTransformerData(NamedTuple):
    updates: Any
    params: Any


class GradientTransformer(pax.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, updates, params):
        return updates

    def build_optimizer(self) -> GradientTransformOptimizer:
        return GradientTransformOptimizer(self)


class Scaler(GradientTransformer):
    def __init__(self, params, scale: float):
        super().__init__()
        self.scale = scale

    def __call__(self, updates, params=None) -> GradientTransformerData:
        return jax.tree_map(lambda u: u * self.scale, updates)


def scale(scale: float):
    return lambda params: Scaler(params=params, scale=scale)


class Clipper(GradientTransformer):
    def __init__(self, params, max_delta):
        super().__init__()
        self.max_delta = max_delta

    def __call__(self, updates, params=None):
        return jax.tree_map(
            lambda u: jnp.clip(u, a_min=-self.max_delta, a_max=self.max_delta), updates
        )


def clip(max_delta: float):
    return lambda params: Clipper(params=params, max_delta=max_delta)


class ClipByGlobalNorm(GradientTransformer):
    def __init__(self, params, global_norm: float):
        super().__init__()
        self.global_norm = global_norm

    def __call__(self, updates, params=None):
        leaves = jax.tree_leaves(updates)
        leaves = jax.tree_map(lambda x: jnp.sum(jnp.square(x)), leaves)
        norm = jnp.sqrt(jnp.sum(jnp.stack(leaves)))
        scale = jnp.clip(self.global_norm / norm, a_max=1.0)
        return jax.tree_map(lambda x: x * scale, updates)


def clip_by_global_norm(global_norm: float):
    lambda params: ClipByGlobalNorm(params=params, global_norm=global_norm)


class Chainer(GradientTransformer):
    transformers: Sequence[GradientTransformer]

    def __init__(self, transformers: Sequence[GradientTransformer]):
        super().__init__()
        self.register_module_subtree("transformers", transformers)

    def __call__(self, updates, params):
        for f in self.transformers:
            updates = f(updates=updates, params=params)

        return updates


def chain(*transformers: Iterable[GradientTransformer]):
    def _chain(params):
        mods = []
        for f in transformers:
            mod = f(params)
            mods.append(mod)
        return Chainer(mods)

    return _chain
