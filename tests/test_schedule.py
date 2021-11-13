import jax
import numpy as np
import pax

import opax


def test_opax_schedule_sgd():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.sgd(scheduler)(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params)
    params = opax.apply_updates(params, updates)


def test_opax_schedule_adam():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adam(scheduler)(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_opax_schedule_adamw():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adamw(scheduler)(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)


def test_opax_schedule_adamw_lr():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adamw(scheduler)(model.parameters())
    params = model.parameters()
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)
    np.testing.assert_almost_equal(
        opt[-1].learning_rate,
        0.9999307,
    )


def test_opax_schedule_adamw_lr_jaxpr():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adamw(scheduler)(model.parameters())
    params = model.parameters()

    def step(opt, a, b):
        opt, updates = pax.module_and_value(opt)(params, params)
        p = opax.apply_updates(params, updates)
        return p, opt

    print(jax.make_jaxpr(step)(opt, params, params))
    opt, updates = pax.module_and_value(opt)(params, params)
    params = opax.apply_updates(params, updates)
    np.testing.assert_almost_equal(
        opt[-1].learning_rate,
        0.9999307,
    )
