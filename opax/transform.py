"""Gradient Transformations."""

from typing import Any, Callable, Optional, Sequence, TypeVar

import jax
import jax.numpy as jnp
import pax

T = TypeVar("T")


class GradientTransformation(pax.StateModule):
    def __init__(self, params=None):
        super().__init__()

    def __call__(self, updates, params=None):
        raise NotImplementedError("A subclass must implement this method")


def identity():
    class Identity(GradientTransformation):
        def __call__(self, updates, params=None):
            return updates

    return Identity


def scale(scale: float):
    class Scale(GradientTransformation):
        def __call__(self, updates, params=None):
            del params
            return jax.tree_util.tree_map(lambda u: u * scale, updates)

    return Scale


def scale_by_schedule(schedule_fn: Callable[[jnp.ndarray], jnp.ndarray]):
    class ScaleBySchedule(GradientTransformation):
        count: jnp.ndarray
        learning_rate: jnp.ndarray

        def __init__(self, params):
            super().__init__(params=params)
            self.schedule_fn = schedule_fn
            self.count = jnp.array(0, dtype=jnp.int32)
            self.learning_rate = self.schedule_fn(self.count)

        def __call__(self, updates, params=None):
            del params
            self.count = self.count + 1
            self.learning_rate = self.schedule_fn(self.count)
            return jax.tree_util.tree_map(lambda u: u * self.learning_rate, updates)

    return ScaleBySchedule


def scale_by_stddev(
    decay_rate: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
):
    class ScaleByStddev(GradientTransformation):
        def __init__(self, params):
            super().__init__(params=params)
            self.mu = jax.tree_util.tree_map(jnp.zeros_like, params)
            self.nu = jax.tree_util.tree_map(
                lambda x: jnp.full_like(x, initial_scale), params
            )

        def __call__(self, updates, params=None):
            del params
            self.mu = _update_moment(updates, self.mu, decay_rate, order=1)
            self.nu = _update_moment(updates, self.nu, decay_rate, order=2)

            updates = jax.tree_util.tree_map(
                lambda g, m, n: g * jax.lax.rsqrt(n - jnp.square(m) + eps),
                updates,
                self.mu,
                self.nu,
            )

            return updates

    return ScaleByStddev


def scale_by_rms(
    decay_rate: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
):
    class ScaleByRms(GradientTransformation):
        def __init__(self, params):
            super().__init__(params=params)
            self.nu = jax.tree_util.tree_map(
                lambda x: jnp.full_like(x, initial_scale), params
            )

        def __call__(self, updates, params=None):
            del params
            self.nu = _update_moment(updates, self.nu, decay_rate, order=2)
            updates = jax.tree_util.tree_map(
                lambda g, n: g * jax.lax.rsqrt(n + eps),
                updates,
                self.nu,
            )
            return updates

    return ScaleByRms


def clip(max_delta: float):
    class Clip(GradientTransformation):
        def __call__(self, updates, params=None):
            del params
            return jax.tree_util.tree_map(
                lambda u: jnp.clip(u, a_min=-max_delta, a_max=max_delta), updates
            )

    return Clip


def _global_norm(x):
    """
    Compute the global norm of a pytree
    """
    leaves = jax.tree_util.tree_leaves(x)
    leaves = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), leaves)
    return jnp.sqrt(jnp.sum(jnp.stack(leaves)))


def clip_by_global_norm(max_global_norm: float):
    class ClipByGlobalNorm(GradientTransformation):
        global_norm: jnp.ndarray  # for logging purposes.

        def __init__(self, params):
            super().__init__(params=params)
            self.global_norm = jnp.array(0.0)

        def __call__(self, updates, params=None):
            del params
            self.global_norm = _global_norm(updates)
            scale = jnp.clip(max_global_norm / self.global_norm, a_max=1.0)
            return jax.tree_util.tree_map(lambda x: x * scale, updates)

    return ClipByGlobalNorm


def trace(decay_rate):
    class Trace(GradientTransformation):
        trace: Any

        def __init__(self, params):
            super().__init__()

            self.trace = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

        def __call__(self, updates, params=None):
            self.trace = jax.tree_util.tree_map(
                lambda u, t: u + t * decay_rate, updates, self.trace
            )
            return self.trace

    return Trace


def add_decayed_weights(weight_decay: float = 0.0):
    class AddDecayedWeights(GradientTransformation):
        def __init__(self, params):
            super().__init__(params=params)

        def __call__(self, updates, params=None):
            assert params is not None, "expecting params argument"

            updates = jax.tree_util.tree_map(
                lambda g, p: g + weight_decay * p, updates, params
            )
            return updates

    return AddDecayedWeights


# source: https://github.com/deepmind/optax/blob/3f42a614096a7cd778e8cab15fd55e4766f47b53/optax/_src/transform.py#L78
def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order-th` moment."""
    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments
    )


# source: https://github.com/deepmind/optax/blob/3f42a614096a7cd778e8cab15fd55e4766f47b53/optax/_src/transform.py#L84
def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_util.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def ema(decay_rate: float, debias: bool = True):
    class EMA(GradientTransformation):
        ema: Any
        count: jnp.ndarray

        def __init__(self, params):
            super().__init__()
            self.count = jnp.array(0, dtype=jnp.int32)
            self.ema = jax.tree_util.tree_map(jnp.zeros_like, params)

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

            self.mu = jax.tree_util.tree_map(jnp.zeros_like, params)
            self.nu = jax.tree_util.tree_map(jnp.zeros_like, params)
            self.count = jnp.array(0, dtype=jnp.int32)

        def __call__(self, updates, params=None):
            del params
            self.mu = _update_moment(updates, self.mu, b1, order=1)
            self.nu = _update_moment(updates, self.nu, b2, order=2)
            self.count = self.count + 1
            mu_hat = _bias_correction(self.mu, b1, self.count)
            nu_hat = _bias_correction(self.nu, b2, self.count)

            updates = jax.tree_util.tree_map(
                lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
            )
            return updates

    return ScaleByAdam


def differentially_private_aggregate(
    l2_norm_clip: float, noise_multiplier: float, seed: int
):
    """
    Return an differentially private optimizer class
    """

    class DifferentiallyPrivateAggreate(GradientTransformation):
        """
        DP optimizer which adds noise to the gradients.
        """

        rng_key: jnp.ndarray
        noise_std = l2_norm_clip * noise_multiplier

        def __init__(self, params):
            super().__init__(params)
            self.rng_key = jax.random.PRNGKey(seed=seed)

        def __call__(self, updates, params=None):
            del params
            grads_flat, grads_treedef = jax.tree_util.tree_flatten(updates)
            bsize = grads_flat[0].shape[0]

            if any(g.ndim == 0 or bsize != g.shape[0] for g in grads_flat):
                raise ValueError(
                    "Unlike other transforms, `differentially_private_aggregate` expects"
                    " `updates` to have a batch dimension in the 0th axis. That is, this"
                    " function expects per-example gradients as input."
                )

            self.rng_key, *rngs = jax.random.split(self.rng_key, len(grads_flat) + 1)
            global_grad_norms = jax.vmap(_global_norm)(grads_flat)
            divisors = jnp.maximum(global_grad_norms / l2_norm_clip, 1.0)
            clipped = [(jnp.moveaxis(g, 0, -1) / divisors).sum(-1) for g in grads_flat]
            noised = [
                (g + self.noise_std * jax.random.normal(r, g.shape, g.dtype)) / bsize
                for g, r in zip(clipped, rngs)
            ]
            updates = jax.tree_util.tree_unflatten(grads_treedef, noised)
            return updates

    return DifferentiallyPrivateAggreate


def chain(*fs: Callable[[Any], GradientTransformation]):
    class Chain(GradientTransformation):
        transforms: Sequence[GradientTransformation]
        flatten: bool
        strict: Optional[bool]

        def __init__(self, params, flatten: bool = False, strict=None):
            """Create a chain of gradient transformations."""
            super().__init__()
            if strict is True and flatten is True:
                raise ValueError("Cannot set `flatten=True` in strict mode.")
            self.flatten = flatten
            self.strict = strict
            if flatten or strict is False:
                leaves = jax.tree_util.tree_leaves(params)
                self.transforms = [f(leaves) for f in fs]
            else:
                self.transforms = [f(params) for f in fs]

        @classmethod
        def init(cls, params, flatten: bool = False, strict=None):
            """Initialize gradient transformations.

            Arguments:
                params: trainable parameters.
                flatten: flatten trainable parameters to a list for faster speed.
                strict: require tree structures to be matched.
            """
            return cls(params=params, flatten=flatten, strict=strict)

        def __call__(self, updates, params=None):
            if self.flatten or self.strict is False:
                updates_leaves, updates_treedef = jax.tree_util.tree_flatten(updates)
                params_leaves, params_treedef = jax.tree_util.tree_flatten(params)

                for f in self.transforms:
                    updates_leaves = f(updates=updates_leaves, params=params_leaves)

                if self.strict is False and params is not None:
                    updates_treedef = params_treedef
                updates = jax.tree_util.tree_unflatten(updates_treedef, updates_leaves)
            else:
                for f in self.transforms:
                    updates = f(updates=updates, params=params)

            return updates

        def __getitem__(self, index: int):
            return self.transforms[index]

    return Chain
