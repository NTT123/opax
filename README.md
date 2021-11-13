# opax

`opax` is an optimizer library for JAX. It is a reimplementation of [optax] using PAX's stateful [module](https://github.com/ntt123/pax).

## Installation

To install the latest version:

```sh
pip3 install git+https://github.com/ntt123/opax.git
```

## Getting started

To create an optimizer:

```python
import opax
optimizer = opax.adam(1e-4).init(parameters)
```

**Note**: ``parameters`` is a pytree of trainable parameters.

To update parameters:

```python
updates, optimizer = opax.transform_gradients(gradients, optimizer, parameters)
parameters = opax.apply_updates(parameters, updates)
```

**Note**: ``gradients`` has the same `treedef` as `parameters`.

## The ``opax.chain`` function

`opax` follows `optax`'s philosophy in combining `GradientTransformation`'s together with ``opax.chain`` function:

```python
optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.scale_by_adam(),
    opax.scale(1e-4),
).init(parameters)
```

## Learning rate schedule

`opax` supports learning rate scheduling with `opax.scale_by_schedule`.

```python
def staircase_schedule_fn(step: jnp.ndarray):
    p = jnp.floor(step.astype(jnp.float32) / 1000)
    return jnp.exp2(-p)

optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.scale_by_adam(),
    opax.scale_by_schedule(staircase_schedule_fn),
).init(parameters)
```


## Utilities

To get the current *global norm*:

```python
print(optimizer[0].global_norm)
```

**Note**: ``global_norm`` is a property of `ClipByGlobalNorm` class.


To get the current *learning rate*:

```python
print(optimizer[-1].learning_rate)
```

**Note**: ``learning_rate`` is a property of `ScaleBySchedule` class.


[optax]: https://optax.readthedocs.io/en/latest/
