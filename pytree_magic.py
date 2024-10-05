import jax.numpy as jnp
import numpy as np
import jax
import time

def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
            biases = np.ones(shape=(n_out,)))
        )
    return params

def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
    #return jnp.mean((forward(params, x) - y) ** 2)
    soft = jax.nn.log_softmax(forward(params, x))
    return -jnp.mean((y * soft[:, 1]) + (1-y) * soft[:, 0])

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)

    return jax.tree_util.tree_map(
        lambda p, g: p - LR * g, params, grads
    )

def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(forward(params, x), axis=-1) == y)

toy_x = np.concatenate([np.random.normal([1, 1], [2, 2], size=(500, 2)), np.random.normal([0, 0], [1, 1], size=(1000, 2))])
toy_y = np.concatenate([np.ones(500), np.zeros(1000)])

params = init_mlp_params([2, 8, 16, 2])

LR = 0.01
EPOCHS = 1000
print(f"Initial loss: {loss_fn(params, toy_x, toy_y):.3f}\tAccuracy: {accuracy(params, toy_x, toy_y) * 100:.2f}%")
start_time = time.time()
for i in range(EPOCHS):
    params = update(params, toy_x, toy_y)

    if (i+1) % (EPOCHS//10) == 0:
        print(f"{i} epoch loss: {loss_fn(params, toy_x, toy_y):.3f}\tAccuracy: {accuracy(params, toy_x, toy_y) * 100:.2f}%")
end_time = time.time()

print(f"Time: {(end_time - start_time)*1000:.4f}ms")