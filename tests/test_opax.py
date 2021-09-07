import opax
import pax


def test_opax_1():
    model = pax.nn.Linear(3, 3)
    learning_rate = 1e-3
    momentum = 0.9
    opt = opax.from_chain(
        opax.scale(learning_rate),
        opax.trace(momentum),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)
