import torch

from model import RMSNorm


def test_RMSNorm():
    config = {
        "batch_size": 5,
        "context_window": 11,
        "d_model": 13,
    }
    batch = torch.randn(
        (config["batch_size"], config["context_window"], config["d_model"])
    )
    m = RMSNorm((config["context_window"], config["d_model"]))
    g = m(batch)
    assert g.shape == torch.Size([5, 11, 13])
