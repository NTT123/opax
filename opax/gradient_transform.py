"""Gradient Transformations."""

from typing import Any, Callable, Sequence, Type

import jax
import jax.numpy as jnp
import pax


class GradientTransformation(pax.Module):
    def __init__(self, params=None):
        super().__init__()

    def __call__(self, updates, params=None):
        raise NotImplementedError("A subclass must implement this method")

    def step(self, grads, params):
        """An optimizing step.

        First, transform gradients
        Second, apply updates to parameters.
        """
        updates = self(grads, params)
        return jax.tree_map(lambda p, u: p - u, params, updates)


def scale(scale: float) -> Type[GradientTransformation]:
    class Scale(GradientTransformation):
        def __call__(self, updates, params=None):
            del params
            return jax.tree_map(lambda u: u * scale, updates)

    return Scale


def clip(max_delta: float) -> Type[GradientTransformation]:
    class Clip(GradientTransformation):
        def __call__(self, updates, params=None):
            del params
            return jax.tree_map(
                lambda u: jnp.clip(u, a_min=-max_delta, a_max=max_delta), updates
            )

    return Clip


def clip_by_global_norm(global_norm: float) -> Type[GradientTransformation]:
    class ClipByGlobalNorm(GradientTransformation):
        def __call__(self, updates, params=None):
            leaves = jax.tree_leaves(updates)
            leaves = jax.tree_map(lambda x: jnp.sum(jnp.square(x)), leaves)
            norm = jnp.sqrt(jnp.sum(jnp.stack(leaves)))
            scale = jnp.clip(global_norm / norm, a_max=1.0)
            return jax.tree_map(lambda x: x * scale, updates)

    return ClipByGlobalNorm


def trace(decay_rate) -> Type[GradientTransformation]:
    class Trace(GradientTransformation):
        trace: Any

        def __init__(self, params):
            super().__init__()

            self.register_state_subtree(
                "trace", jax.tree_map(lambda x: jnp.zeros_like(x), params)
            )

        def __call__(self, updates, params=None):
            return jax.tree_map(lambda u, t: u + t * decay_rate, updates, self.trace)

    return Trace


# source: https://github.com/deepmind/optax/blob/3f42a614096a7cd778e8cab15fd55e4766f47b53/optax/_src/transform.py#L78
def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_multimap(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments
    )


def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def ema(decay_rate: float, debias: bool = True) -> Type[GradientTransformation]:
    class EMA(GradientTransformation):
        ema: Any

        def __init__(self, params):
            super().__init__()
            self.register_state("count", jnp.array(0, dtype=jnp.int32))
            self.register_state_subtree(
                "ema", jax.tree_map(lambda x: jnp.zeros_like(x), params)
            )

        def __call__(self, updates, params=None):
            del params
            self.ema = _update_moment(updates, self.ema, decay_rate, order=1)
            self.count = self.count + 1
            if debias:
                self.ema = _bias_correction(self.ema, decay_rate, self.count)
            return self.ema

    return EMA


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
):
    class ScaleByAdam(GradientTransformation):
        mu: Any
        nu: Any
        count: jnp.ndarray

        def __init__(self, params):
            super().__init__(params)

            self.register_state_subtree(
                "mu", jax.tree_map(lambda x: jnp.zeros_like(x), params)
            )
            self.register_state_subtree(
                "nu", jax.tree_map(lambda x: jnp.zeros_like(x), params)
            )
            self.register_state("count", jnp.array(0, dtype=jnp.int32))

        def __call__(self, updates, params=None):
            del params
            self.mu = _update_moment(updates, self.mu, b1, order=1)
            self.nu = _update_moment(updates, self.nu, b2, order=2)
            self.count = self.count + 1
            mu_hat = _bias_correction(self.mu, b1, self.count)
            nu_hat = _bias_correction(self.nu, b2, self.count)

            updates = jax.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
            )
            return updates

    return ScaleByAdam


def chain(*fs: Callable[[Any], GradientTransformation]) -> Type[GradientTransformation]:
    class Chain(GradientTransformation):
        transforms: Sequence[GradientTransformation]

        def __init__(self, params):
            super().__init__()
            transforms = [f(params) for f in fs]
            self.register_module_subtree("transforms", transforms)

        def __call__(self, updates, params=None):
            for f in self.transforms:
                updates = f(updates=updates, params=params)

            return updates

    return Chain
