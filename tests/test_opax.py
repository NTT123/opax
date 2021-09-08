import jax.numpy as jnp
import numpy as np
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


def test_trace():
    x = jnp.array(1.0)
    t = opax.trace(0.9)(x)
    t(x)
    assert t.trace.item() == 1.0
    t(x * 0.0)
    np.testing.assert_almost_equal(t.trace, 0.9)
    t(x)
    np.testing.assert_almost_equal(t.trace, 0.9 * 0.9 + 1.0)


def test_opax_global_norm():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale(1e-3),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)

    assert opt[0].global_norm >= 0.0
