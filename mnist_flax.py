#https://flax.readthedocs.io/en/v0.5.3/notebooks/annotated_mnist.html

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import numpy as np
import optax
import tensorflow_datasets as tfds

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

def get_datasets():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

def create_train_state(rng, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = CNN().apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics

@jax.jit
def eval_step(params, batch):
    logits = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])

def train_epoch(state, train_ds, batch_size, epoch, rng):
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)
    
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print(f"Train epoch: {epoch}: loss: {epoch_metrics_np['loss']:.4f}, accuracy: {epoch_metrics_np['accuracy']*100:.2f}")
    return state

def eval_model(params, test_ds):
    metrics = eval_step(params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


train_ds, test_ds = get_datasets()
train_set_size, test_set_size = 5000, 5000
train_ds['image'] = train_ds['image'][:train_set_size]
train_ds['label'] = train_ds['label'][:train_set_size]
test_ds['image'] = test_ds['image'][:test_set_size]
test_ds['label'] = test_ds['label'][:test_set_size]
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

learning_rate = 0.1
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)
del init_rng

num_epochs = 10
batch_size = 32

for epoch in range(1, num_epochs+1):
    rng, input_rng = jax.random.split(rng)
    state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
    test_loss, test_accuracy = eval_model(state.params, test_ds)
    print(f"Test epoch: {epoch}: loss: {test_loss:.4f}, accuracy: {test_accuracy*100:.2f}")