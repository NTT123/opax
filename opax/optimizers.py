from typing import Any, Callable, Sequence

from .base_optimizer import Optimizer
from .gradient_transformers import GradientTransformation, chain, scale, trace


def from_chain(*transforms: Sequence[Callable[[Any], GradientTransformation]]):
    class ChainOptimizer(Optimizer):
        def __init__(self, params):
            super().__init__(params)
            self.gradient_transformer = chain(*transforms)(params)

    return ChainOptimizer


def sgd(learning_rate: float = 1e-2, momentum: float = 0.9):
    return from_chain(scale(learning_rate), trace(momentum))
