import jax
import jax.numpy as jnp
import numpy as np
import opax
import pax
import pytest


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


def test_all_finite_predicate():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.adam(1e-3),
    )(model.parameters())
    params = model.parameters()
    params = opt.step(params, params, all_finite=jnp.array(False))
    assert opt[-1][0].count.item() == 0
    params = opt.step(params, params, all_finite=jnp.array(True))
    assert opt[-1][0].count.item() == 1
    params = opt.step(params, params, all_finite=jnp.array(False))
    assert opt[-1][0].count.item() == 1


def test_train_1():
    net = pax.nn.Linear(1, 1)

    def loss_fn(params, model, inputs) -> pax.utils.LossFnOutput:
        loss = jnp.mean(jnp.square(model.update(params)(inputs[0]) - inputs[1]))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    for i in range(10):
        loss, net, opt = update_fn(net, opt, (x, x))


def test_train_2():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.Linear(2, 1),
    )

    def loss_fn(params, model, inputs) -> pax.utils.LossFnOutput:
        loss = jnp.mean(jnp.square(model.update(params)(inputs[0]) - inputs[1]))
        model.modules[-1] = pax.utils.Lambda(jax.nn.relu)
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    with pytest.raises(ValueError):
        for i in range(10):
            loss, net, opt = update_fn(net, opt, (x, x))


def test_train_flatten():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.Linear(2, 1),
    )

    def loss_fn(params, model, inputs) -> pax.utils.LossFnOutput:
        loss = jnp.mean(jnp.square(model.update(params)(inputs[0]) - inputs[1]))
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters(), flatten=True)
    for i in range(10):
        loss, net, opt = update_fn(net, opt, (x, x))
