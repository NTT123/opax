import opax
import pax


def test_opax_schedule_sgd():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.sgd(scheduler)(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)


def test_opax_schedule_adam():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adam(scheduler)(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)


def test_opax_schedule_adamw():
    model = pax.nn.Linear(3, 3)
    scheduler = opax.schedule.exponential_decay(1.0, 10_000)
    opt = opax.adamw(scheduler)(model.parameters())
    params = model.parameters()
    params = opt.step(params, params)
