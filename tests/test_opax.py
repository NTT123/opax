import jax
import jax.numpy as jnp
import numpy as np
import opax
import pax
import pytest


def test_opax_1():
    model = pax.Linear(3, 3)
    learning_rate = 1e-3
    momentum = 0.9
    opt = opax.chain(
        opax.trace(momentum),
        opax.scale(learning_rate),
    )(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_opax_sgd():
    model = pax.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_opax_step_sgd():
    model = pax.Linear(3, 3)
    opt = opax.chain(
        opax.sgd(1e-2, 0.9),
    )(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_opax_adam():
    model = pax.Linear(3, 3)
    opt = opax.adam(1e-3)(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_trace():
    x = jnp.array(1.0)
    t = opax.trace(0.9)(x)
    t, _ = pax.module_and_value(t)(x)
    assert t.trace.item() == 1.0
    t, _ = pax.module_and_value(t)(x * 0.0)
    np.testing.assert_almost_equal(t.trace, 0.9)
    t, _ = pax.module_and_value(t)(x)
    np.testing.assert_almost_equal(t.trace, 0.9 * 0.9 + 1.0)


def test_opax_global_norm():
    model = pax.Linear(3, 3)
    opt = opax.chain(
        opax.clip_by_global_norm(1.0),
        opax.scale(1e-3),
    )(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)

    assert opt[0].global_norm >= 0.0


# def test_all_finite_predicate():
#     model = pax.Linear(3, 3)
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


def test_train_1():
    net = pax.Linear(1, 1)

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        optimizer, updates = pax.module_and_value(optimizer)(grads)
        model = model.update_parameters(opax.apply_updates(model.parameters(), updates))
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    for _ in range(10):
        net, opt, _ = update_fn(net, opt, (x, x))


def test_train_2():
    net = pax.Sequential(
        pax.Linear(1, 2),
        pax.Linear(2, 1),
    )

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        model = model.set(-1, pax.Lambda(jax.nn.relu))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters())
    with pytest.raises((AssertionError, ValueError)):
        for _ in range(10):
            net, opt, _ = update_fn(net, opt, (x, x))


def test_train_flatten():
    net = pax.Sequential(
        pax.Linear(1, 2),
        pax.Linear(2, 1),
    )

    def loss_fn(model, inputs):
        loss = jnp.mean(jnp.square(model(inputs[0]) - inputs[1]))
        return loss, model

    def update_fn(model, optimizer, inputs):
        (loss, model), grads = jax.value_and_grad(loss_fn, has_aux=True)(model, inputs)
        model, optimizer = opax.apply_gradients(model, optimizer, grads=grads)
        return model, optimizer, loss

    x = jnp.zeros((1, 1))
    opt = opax.adam()(net.parameters(), flatten=True)
    for _ in range(10):
        net, opt, _ = update_fn(net, opt, (x, x))
