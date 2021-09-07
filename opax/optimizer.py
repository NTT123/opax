import jax
import pax
from typing import Any, Callable, Sequence


class GradientTransformOptimizer(pax.Module):
    def __init__(self, gradient_transformer):
        super().__init__()
        self.gradient_transformer = gradient_transformer

    def step(self, params, gradients):
        updates = self.gradient_transformer(gradients, params)
        new_params = jax.tree_map(lambda u, p: p - u, updates, params)
        return new_params
