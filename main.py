import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

from model import SimpleBrokenModel, SimpleModel
from utils import get_batches


lines = open("./input.txt", "r").read()

vocab = sorted(list(set(lines)))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

assert len(vocab) == 65  # Individual characters

print(lines[:30])


# simple tokenization by characters
def encode(s):
    return [stoi[ch] for ch in s]


def decode(l):
    return "".join([itos[i] for i in l])


print("vocab size:", len(vocab))
decode(encode("hello"))

MASTER_CONFIG = {
    "vocab_size": len(vocab),
}

dataset = torch.tensor(encode(lines), dtype=torch.int8)
print(dataset.shape)
assert dataset.shape == torch.Size([1115393])




MASTER_CONFIG.update({"batch_size": 32, "context_window": 16})

xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

print([(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))])


@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(
                dataset, split, config["batch_size"], config["context_window"]
            )
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


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


def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()

        xs, ys = get_batches(
            dataset, "train", config["batch_size"], config["context_window"]
        )
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config["log_interval"] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(
                    f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}"
                )
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("validation loss: ", losses[-1]["val"])
    return pd.DataFrame(losses)


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


loss_plot = train(model, optimizer)
loss_plot.plot()
plt.show()

# ------------------------------
model = SimpleModel(MASTER_CONFIG)
xs, ys = get_batches(
    dataset, "train", MASTER_CONFIG["batch_size"], MASTER_CONFIG["context_window"]
)

logits, loss = model(xs, ys)
optimizer = torch.optim.Adam(model.parameters())
loss_plot = train(model, optimizer)
loss_plot.plot()
plt.show()

print(generate(model))
