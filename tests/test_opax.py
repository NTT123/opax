import jax
import jax.numpy as jnp
import numpy as np
import pax
import pytest

import opax


@pax.pure
def test_opax_1():
    model = pax.nn.Linear(3, 3)
    learning_rate = 1e-3
    momentum = 0.9
    opt = opax.chain(
        opax.trace(momentum),
        opax.scale(learning_rate),
    )(model.parameters())
    params = model.parameters()
    updates = opt(params, params)
    params = opax.apply_updates(params, updates)


@pax.pure
def test_opax_sgd():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    updates = opt(params, params)
    params = opax.apply_updates(params, updates)


@pax.pure
def test_opax_step_sgd():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    updates = opt(params, params)
    params = opax.apply_updates(params, updates)


@pax.pure
def test_opax_adam():
    model = pax.nn.Linear(3, 3)
    opt = opax.adam(1e-3)(model.parameters())
    params = model.parameters()
    updates = opt(params, params)
    params = opax.apply_updates(params, updates)


@pax.pure
def test_trace():
    x = jnp.array(1.0)
    t = opax.trace(0.9)(x)
    t(x)
    assert t.trace.item() == 1.0
    t(x * 0.0)
    np.testing.assert_almost_equal(t.trace, 0.9)
    t(x)
    np.testing.assert_almost_equal(t.trace, 0.9 * 0.9 + 1.0)


@pax.pure
def test_opax_global_norm():
    model = pax.nn.Linear(3, 3)
    opt = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale(1e-3),
    )(model.parameters())
    params = model.parameters()
    updates = opt(params, params)
    params = opax.apply_updates(params, updates)

    assert opt[0].global_norm >= 0.0


# def test_all_finite_predicate():
#     model = pax.nn.Linear(3, 3)
#     opt = opax.chain(
#         opax.clip_by_global_norm(1.0),
#         opax.adam(1e-3),
#     )(model.parameters())
#     params = model.parameters()
#     params = opt.step(params, params, all_finite=jnp.array(False))
#     assert opt[-1][0].count.item() == 0
#     params = opt.step(params, params, all_finite=jnp.array(True))
#     assert opt[-1][0].count.item() == 1
#     params = opt.step(params, params, all_finite=jnp.array(False))
#     assert opt[-1][0].count.item() == 1


@pax.pure
def test_train_1():
    net = pax.nn.Linear(1, 1)

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        updates = optimizer(grads)
        model = model.update_parameters(opax.apply_updates(model.parameters(), updates))
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    for _ in range(10):
        net, opt, _ = update_fn(net, opt, (x, x))


@pax.pure
def test_train_2():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.Linear(2, 1),
    )

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        model = model.set(-1, pax.nn.Lambda(jax.nn.relu))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        model, optimizer = pax.apply_gradients(model, optimizer, grads=grads)
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    with pytest.raises((AssertionError, ValueError)):
        for _ in range(10):
            net, opt, _ = update_fn(net, opt, (x, x))


def test_train_flatten():
    net = pax.nn.Sequential(
        pax.nn.Linear(1, 2),
        pax.nn.Linear(2, 1),
    )

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        model, optimizer = pax.apply_gradients(model, optimizer, grads=grads)
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters(), flatten=True)
    for _ in range(10):
        net, opt, _ = update_fn(net, opt, (x, x))
