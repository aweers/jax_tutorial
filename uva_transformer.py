# Following: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html

## Standard libraries
import os
import numpy as np
import math
import json
from functools import partial

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib

import seaborn as sns

## tqdm for loading bars
from tqdm.auto import tqdm

## JAX
import jax
import jax.numpy as jnp
from jax import random

## Flax (NN in JAX)
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

import urllib.request
from urllib.error import HTTPError


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../../saved_models/tutorial6_jax"

print("Device:", jax.devices()[0])

matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
# Seeding for random operations
main_rng = random.PRNGKey(42)

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial6/"
# Files to download
pretrained_files = ["ReverseTask.ckpt", "SetAnomalyTask.ckpt"]

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                e,
            )


# Transformer implementation from scratch
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -jnp.inf, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention


seq_len, d_k = 3, 2
main_rng, rand1 = random.split(main_rng)
qkv = random.normal(rand1, (3, seq_len, d_k))
q, k, v = qkv[0], qkv[1], qkv[2]
mask = jnp.arange(seq_len) <= jnp.arange(seq_len)[:, None]
values, attention = scaled_dot_product(q, k, v, mask=mask)

print(f"Q\n{q}")
print(f"K\n{k}")
print(f"V\n{v}")
print(f"Values\n{values}")
print(f"Attention\n{attention}")


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )
        self.o_proj = nn.Dense(
            self.embed_dim, kernel_init=nn.initializers.xavier_uniform(), use_bias=False
        )

    def __call__(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)

        # Seperate qkv
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # batch x heads x seq_len x dims
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # batch x seq_len x head x dims
        values = values.reshape(batch_size, seq_len, embed_dim)
        o = self.o_proj(values)

        return o, attention


## Test MultiheadAttention implementation
# Example features as input
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 128))
# Create attention
mh_attn = MultiheadAttention(embed_dim=128, num_heads=4)
# Initialize parameters of attention with random key and inputs
main_rng, init_rng = random.split(main_rng)
params = mh_attn.init(init_rng, x)["params"]
# Apply attention with parameters on the inputs
out, attn = mh_attn.apply({"params": params}, x)
print("Out", out.shape, "Attention", attn.shape)

del mh_attn, params


class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_ff: int
    dropout_p: float

    def setup(self):
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )

        # two layer MLP
        self.linear = [
            nn.Dense(self.dim_ff),
            nn.Dropout(self.dropout_p),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_p)

    def __call__(self, x, mask=None, train=True):
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        linear_out = x
        for layer in self.linear:
            linear_out = (
                layer(linear_out)
                if not isinstance(layer, nn.Dropout)
                else layer(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


## Test EncoderBlock implementation
# Example features as input
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 128))
# Create encoder block
encblock = EncoderBlock(input_dim=128, num_heads=4, dim_ff=512, dropout_p=0.1)
# Initialize parameters of encoder block with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encblock.init(
    {"params": init_rng, "dropout": dropout_init_rng}, x, train=True
)["params"]
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
out = encblock.apply(
    {"params": params}, x, train=True, rngs={"dropout": dropout_apply_rng}
)
print("Out", out.shape)

del encblock, params


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_ff: int
    dropout_p: float

    def setup(self):
        self.layers = [
            EncoderBlock(self.input_dim, self.num_heads, self.dim_ff, self.dropout_p)
            for _ in range(self.num_layers)
        ]

    def __call__(self, x, mask=None, train=True):
        for layer in self.layers:
            x = layer(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = layer(x, mask=mask, train=train)

        return attention_maps


## Test TransformerEncoder implementation
# Example features as input
main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (3, 16, 128))
# Create Transformer encoder
transenc = TransformerEncoder(
    num_layers=5, input_dim=128, num_heads=4, dim_ff=256, dropout_p=0.15
)
# Initialize parameters of transformer with random key and inputs
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = transenc.init(
    {"params": init_rng, "dropout": dropout_init_rng}, x, train=True
)["params"]
# Apply transformer with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward
main_rng, dropout_apply_rng = random.split(main_rng)
# Instead of passing params and rngs every time to a function call, we can bind them to the module
binded_mod = transenc.bind({"params": params}, rngs={"dropout": dropout_apply_rng})
out = binded_mod(x, train=True)
print("Out", out.shape)
attn_maps = binded_mod.get_attention_maps(x, train=True)
print("Attention maps", len(attn_maps), attn_maps[0].shape)

del transenc, binded_mod, params
