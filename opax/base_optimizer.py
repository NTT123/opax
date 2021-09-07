from typing import Any, Callable, Sequence

import jax
import pax


class Optimizer(pax.Module):
    def __init__(self, params):
        super().__init__()
        self.gradient_transformer = lambda g, p=None: g

    def step(self, params, gradients):
        updates = self.gradient_transformer(gradients, params)
        new_params = jax.tree_map(lambda u, p: p - u, updates, params)
        return new_params
