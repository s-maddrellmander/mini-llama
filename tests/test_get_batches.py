import pytest
import torch

from utils import get_batches

# Sample placeholder for MASTER_CONFIG based on context provided.
MASTER_CONFIG = {"batch_size": 10, "context_window": 5}
LARGER_DATA_SIZE = 200


def decode(tensor):
    # Placeholder decode function. Your actual decode function might vary.
    return "".join(map(chr, tensor))


def test_get_batches_on_val_data():
    data = torch.arange(0, LARGER_DATA_SIZE)  # Larger mock dataset
    smaller_context_window = 5
    x, y = get_batches(data, "val", MASTER_CONFIG["batch_size"], smaller_context_window)

    assert x.shape == (MASTER_CONFIG["batch_size"], smaller_context_window)
    assert y.shape == (MASTER_CONFIG["batch_size"], smaller_context_window)

    for xi, yi in zip(x, y):
        assert decode(xi)[1:] == decode(yi)[:-1]


def test_get_batches_on_test_data():
    data = torch.arange(0, LARGER_DATA_SIZE)  # Larger mock dataset
    smaller_context_window = 5
    x, y = get_batches(
        data, "test", MASTER_CONFIG["batch_size"], smaller_context_window
    )

    assert x.shape == (MASTER_CONFIG["batch_size"], smaller_context_window)
    assert y.shape == (MASTER_CONFIG["batch_size"], smaller_context_window)

    for xi, yi in zip(x, y):
        assert decode(xi)[1:] == decode(yi)[:-1]


def test_get_batches_with_incorrect_split():
    data = torch.arange(97, 123)
    with pytest.raises(ValueError):  # Raise ValueError for invalid splits
        get_batches(
            data,
            "invalid_split",
            MASTER_CONFIG["batch_size"],
            MASTER_CONFIG["context_window"],
        )


def test_get_batches_with_large_context_window():
    data = torch.arange(97, 123)
    with pytest.raises(
        ValueError
    ):  # Assuming an error is raised when context_window is too large
        get_batches(data, "train", MASTER_CONFIG["batch_size"], 1000)
