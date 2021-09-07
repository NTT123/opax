import opax
import pax


def test_opax_1():
    model = pax.nn.Linear(3, 3)
    learning_rate = 1e-3
    momentum = 0.9
    opt = opax.chain(
        opax.trace(momentum),
        opax.scale(learning_rate),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)


def test_opax_sgd():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)


def test_opax_step_sgd():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)


def test_opax_adam():
    model = pax.nn.Linear(3, 3)
    opt = opax.adam(1e-3)(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)
