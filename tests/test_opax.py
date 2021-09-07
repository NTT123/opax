import opax
import pax


def test_opax_1():
    model = pax.nn.Linear(3, 3)
    learning_rate = 1e-3
    opt = opax.chain_optimizer(
        opax.scale(learning_rate),
        opax.clip(1.0),
    )(model.parameters())

    params = model.parameters()
    params = opt.step(params, params)
