"""train a handwritten digit classifier."""

from typing import List, Mapping, Tuple

import jax
import jax.numpy as jnp
import opax
import pax
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

Batch = Mapping[str, jnp.ndarray]

# config
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
weight_decay = 1e-4


class ConvNet(pax.Module):
    """ConvNet module."""

    layers: List[Tuple[pax.nn.Conv2D, pax.nn.BatchNorm2D]]
    output: pax.nn.Conv2D

    def __init__(self):
        super().__init__()
        self.layers = []
        for i in range(5):
            conv = pax.nn.Conv2D((1 if i == 0 else 32), 32, 6, padding="VALID")
            bn = pax.nn.BatchNorm2D(32, True, True, 0.9)
            self.layers.append((conv, bn))
        self.output = pax.nn.Conv2D(32, 10, 3, padding="VALID")

    def __call__(self, x: jnp.ndarray):
        for conv, bn in self.layers:
            x = bn(conv(x))
            x = jax.nn.relu(x)
        x = self.output(x)
        return jnp.squeeze(x, (1, 2))


def loss_fn(model: ConvNet, batch: Batch):
    x = batch["image"].astype(jnp.float32) / 255
    target = batch["label"]
    logits = model(x)
    log_pr = jax.nn.log_softmax(logits, axis=-1)
    log_pr = jnp.sum(jax.nn.one_hot(target, log_pr.shape[-1]) * log_pr, axis=-1)
    loss = -jnp.mean(log_pr)
    return loss, (loss, model)


@jax.jit
def test_loss_fn(model: ConvNet, batch: Batch):
    model = model.eval()
    return loss_fn(model, batch)[0]


@jax.jit
def update_fn(model: ConvNet, optimizer: opax.GradientTransformation, batch: Batch):
    grads, (loss, model) = jax.grad(loss_fn, has_aux=True, allow_int=int)(model, batch)
    params = model.parameters()
    updates = optimizer(grads.parameters(), params=params)
    new_params = pax.apply_updates(params, updates=updates)
    model = model.update_parameters(new_params)
    return model, optimizer, loss


net = ConvNet()
print(net.summary())
optimizer = opax.chain(
    opax.clip_by_global_norm(1.0),
    opax.adam(learning_rate=learning_rate),
)(net.parameters())


def load_dataset(split: str):
    """Loads the dataset as a tensorflow dataset."""
    ds = tfds.load("mnist:3.*.*", split=split)
    return ds


train_data = load_dataset("train").shuffle(10 * batch_size).batch(batch_size)
test_data = load_dataset("test").shuffle(10 * batch_size).batch(batch_size)


for epoch in range(0, 10):
    losses, global_norm = 0.0, 0.0
    for batch in tqdm(train_data, desc="train", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        net, optimizer, loss = update_fn(net, optimizer, batch)
        losses = losses + loss
        global_norm = global_norm + optimizer[0].global_norm
    loss = losses / len(train_data)
    global_norm = global_norm / len(train_data)

    test_losses = 0.0
    for batch in tqdm(test_data, desc="eval", leave=False):
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        test_losses = test_losses + test_loss_fn(net, batch)
    test_loss = test_losses / len(test_data)

    print(
        f"[Epoch {epoch}]  train loss {loss:.3f}  test loss {test_loss:.3f}  global norm {global_norm:.3f}"
    )
