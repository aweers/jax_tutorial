# Following https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html

## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
print("Using jax", jax.__version__)

# same syntax as numpy
a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)
b = jnp.arange(6)
print(b)

# arrays are of type DeviceArray
print(b.__class__, type(b))
# could be on CPU, GPU or TPU
print(b.devices())

# move array to cpu
b_cpu = jax.device_get(b)
print(type(b_cpu))
# move array to gpu/metal
b_device = jax.device_put(b_cpu)
print(b_device.devices(), type(b_device))

# jax automatically handles variables on different devices
print(type(b_cpu + b_device))

# see available devices
print(jax.devices())


# Immutable
# no inplace operations allowed
# ie b[0] = 1 raises Exception
# instead
b_new = b.at[0].set(1)
print(b, b_new)


# Random numbers
# "more complex" due to avoidance of side effects
# comes with advantage of greater control
rng = jax.random.PRNGKey(42)

# A non-desirable way of generating pseudo-random numbers...
jax_random_number_1 = jax.random.normal(rng)
jax_random_number_2 = jax.random.normal(rng)
print('JAX - Random number 1:', jax_random_number_1)
print('JAX - Random number 2:', jax_random_number_2)

# Typical random numbers in NumPy
np.random.seed(42)
np_random_number_1 = np.random.normal()
np_random_number_2 = np.random.normal()
print('NumPy - Random number 1:', np_random_number_1)
print('NumPy - Random number 2:', np_random_number_2)

rng, subkey1, subkey2 = jax.random.split(rng, num=3)  # We create 3 new keys
jax_random_number_1 = jax.random.normal(subkey1)
jax_random_number_2 = jax.random.normal(subkey2)
print('JAX new - Random number 1:', jax_random_number_1)
print('JAX new - Random number 2:', jax_random_number_2)



# simple function and some experiments on it
def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.arange(3, dtype=jnp.float32)
print(f"Input: {inp}")
print(f"Output: {simple_graph(inp)}")

# print the jaxpr representation
print(jax.make_jaxpr(simple_graph)(inp))

# automatic differentiation
grad_function = jax.grad(simple_graph)
gradients = grad_function(inp)
print(f"Gradients: {gradients}")

# use jax to get the analytical gradient function (feature by-design)
print(f"Jaxpr representation of gradients: {jax.make_jaxpr(grad_function)(inp)}")

val_grad_function = jax.value_and_grad(simple_graph)
print(f"Value and gradient: {val_grad_function(inp)}")

# Use XLA JIT to compile functions (i.e. remove unneccessary calculations and fuse operations)
jitted_function = jax.jit(simple_graph)

rng, normal_rng = jax.random.split(rng)
large_input = jax.random.normal(normal_rng, (1000,))

# A JIT function is compiled during the first run, since it depends on the shape of the input
# NOTE: Thus it would be necessary to compile a function again, when the input shape changes. 
# This is the main reason why padding is common in jax programs
_ = jitted_function(large_input)

import timeit

def stop_time(stmt, repeat=1000, number=1):
    result = min(timeit.repeat(stmt, repeat=repeat, number=number, globals=globals()))
    if result >= 0.1:
        return f"{result:.3f}s"
    if result >= 0.001:
        return f"{result*1000:2f}ms"
    return f"{result*1000000:2f}µs"

print(f"Timing of simple_graph: {stop_time('simple_graph(large_input).block_until_ready()')}")
print(f"Timing of jitted_function: {stop_time('jitted_function(large_input).block_until_ready()')}")

jitted_grad_function = jax.jit(grad_function)
_ = jitted_grad_function(large_input)

print(f"Timing of grad_function: {stop_time('grad_function(large_input).block_until_ready()', repeat=100)}")
print(f"Timing of jitted_grad_function: {stop_time('jitted_grad_function(large_input).block_until_ready()', repeat=100)}")


# Flax
import flax
from flax import linen as nn

class SimpleClassifier(nn.Module):
    num_hidden: int
    num_outputs: int

    def setup(self):
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x

# use @nn.compact for model definition without redundant code
class SimpleClassifierCompact(nn.Module):
    num_hidden: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x

model = SimpleClassifier(num_hidden=8, num_outputs=1)
print(f"Model: {model}")

rng, inp_rng, init_rng = jax.random.split(rng, 3)
inp = jax.random.normal(inp_rng, (8, 2)) # batchsize 8, 2 inputs
params = model.init(init_rng, inp)
print(params)

output = model.apply(params, inp)
print(f"Model output: {output}")

import torch.utils.data as data

class XORDataset(data.Dataset):
    def __init__(self, size, seed, std=0.1):
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()
    
    def generate_continuous_xor(self):
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int32)

        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

dataset = XORDataset(size=200, seed=42)
print(f"Size of dataset: {len(dataset)}")
print(f"Data point 0: {dataset[0]}")

def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

#visualize_samples(dataset.data, dataset.label)
#plt.show()

# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)

data_inputs, data_labels = next(iter(data_loader))

print(f"Data inputs: {data_inputs.shape}\n{data_inputs}")
print(f"Data labels: {data_labels.shape}\n{data_labels}")

import optax
from flax.training import train_state

optimizer = optax.sgd(learning_rate=0.1)
model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)

    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    acc = (pred_labels == labels).mean()

    return loss, acc

# with jax.jit: 20s / 15 epochs = 1.33s / epoch
# without     : 107s / 2 epochs = 53.5s / epoch (x40.125)!!
@jax.jit
def train_step(state, batch):
    # argnums specifies the location of the input where we want to derive wrt
    # has_aux=True since we have auxilary output (acc)
    grad_fn = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)

    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

train_dataset = XORDataset(size=2500, seed=42)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)

def train_model(state, data_loader, num_epochs=100):
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
    
    return state

trained_model_state = train_model(model_state, train_dataset, num_epochs=15)

test_dataset = XORDataset(size=500, seed=123)
test_data_loader = data.DataLoader(test_dataset,
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   collate_fn=numpy_collate)

def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

eval_model(trained_model_state, test_data_loader)