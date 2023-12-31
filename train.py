import time

import numpy as np
import pandas as pd
import torch

from utils import get_batches


@torch.no_grad()  # don't compute gradients for this function
def evaluate_loss(model, dataset, config=None):
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


def train(model, optimizer, dataset, scheduler=None, config=None, print_logs=False):
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
            x = evaluate_loss(model, dataset, config=config)
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
