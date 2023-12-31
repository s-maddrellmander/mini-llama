import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

from llama import Llama, LlamaBlock
from model import (
    RoPEAttention,
    RoPEAttention_wMask,
    RopeModel,
    RopeModelSwish,
    SimpleBrokenModel,
    SimpleModel,
    SimpleModel_RMS,
)
from train import train
from utils import get_batches, get_dataset


dataset, vocab = get_dataset()


MASTER_CONFIG = {
    "vocab_size": len(vocab),
}


MASTER_CONFIG.update({"batch_size": 32, "context_window": 16})

xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

MASTER_CONFIG.update(
    {
        "d_model": 128,
    }
)
model = SimpleBrokenModel(MASTER_CONFIG)
xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

logits, loss = model(xs, ys)

MASTER_CONFIG.update({"epochs": 1000, "log_interval": 10})

model = SimpleBrokenModel(MASTER_CONFIG)

optimizer = torch.optim.Adam(
    model.parameters(),
)


def generate(model, config=MASTER_CONFIG, max_new_tokens=30):
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config["context_window"] :])
        last_time_step_logits = logits[
            :, -1, :
        ]  # all the batches (1), last time step, all the logits
        p = F.softmax(last_time_step_logits, dim=-1)  # softmax to get probabilities
        idx_next = torch.multinomial(
            p, num_samples=1
        )  # sample from the distribution to get the next token
        idx = torch.cat([idx, idx_next], dim=-1)  # append to the sequence
    return [decode(x) for x in idx.tolist()]


loss_plot = train(model, optimizer, dataset, config=MASTER_CONFIG)
loss_plot.plot()
plt.show()

# ------------------------------
model = SimpleModel(MASTER_CONFIG)
xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
loss_plot = train(model, optimizer, dataset, config=MASTER_CONFIG)
loss_plot.plot()
plt.show()

# print(generate(model))
# ----------------------------

model = SimpleModel_RMS(MASTER_CONFIG)
xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
loss_plot = train(model, optimizer, dataset, config=MASTER_CONFIG)
loss_plot.plot()
plt.show()

# -----------------------------
MASTER_CONFIG.update(
    {
        "n_heads": 8,
    }
)
layer = RoPEAttention(MASTER_CONFIG)
batch = torch.ones(
    (
        MASTER_CONFIG["batch_size"],
        MASTER_CONFIG["context_window"],
        MASTER_CONFIG["d_model"],
    )
)
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy(), interpolation="nearest")
plt.colorbar()
plt.show()

layer = RoPEAttention_wMask(MASTER_CONFIG)
batch = torch.ones(
    (
        MASTER_CONFIG["batch_size"],
        MASTER_CONFIG["context_window"],
        MASTER_CONFIG["d_model"],
    )
)
output, attn_weights = layer(batch, return_attn_weights=True)

plt.imshow(attn_weights[0].detach().numpy())
plt.colorbar()
plt.show()
# ---------------------------------
MASTER_CONFIG.update(
    {
        "epochs": 5000,
        "log_interval": 10,
        "n_heads": 8,
    }
)
model = RopeModelSwish(MASTER_CONFIG)
xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
loss_plot = train(model, optimizer, dataset, config=MASTER_CONFIG)
loss_plot.plot()
plt.show()

# ------------------------------------
MASTER_CONFIG.update(
    {
        "n_layers": 4,
    }
)
llama = Llama(MASTER_CONFIG)
optimizer = torch.optim.Adam(llama.parameters())
train(llama, optimizer, config=MASTER_CONFIG)
