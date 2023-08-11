import torch


def get_batches(data, split, batch_size, context_window, config=None):
    train = data[: int(0.8 * len(data))]
    val = data[int(0.8 * len(data)) : int(0.9 * len(data))]
    test = data[int(0.9 * len(data)) :]

    if split == "train":
        batch_data = train
    elif split == "val":
        batch_data = val
    elif split == "test":
        batch_data = test
    else:
        raise ValueError(
            f"Invalid split '{split}' provided. Expected 'train', 'val', or 'test'."
        )

    if len(batch_data) < context_window:
        raise ValueError(
            f"Data length after split is smaller than the context window. Reduce context_window or use a larger dataset."
        )

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i : i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1 : i + context_window + 1] for i in ix]).long()
    return x, y


def get_dataset():
    lines = open("./input.txt", "r").read()

    vocab = sorted(list(set(lines)))
    itos = {i: ch for i, ch in enumerate(vocab)}
    stoi = {ch: i for i, ch in enumerate(vocab)}

    assert len(vocab) == 65  # Individual characters

    # simple tokenization by characters
    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    dataset = torch.tensor(encode(lines), dtype=torch.int8)
    print(dataset.shape)
    assert dataset.shape == torch.Size([1115393])

    return dataset, vocab
