import torch

from model import RoPEAttention
from rope_encoding import get_rotary_matrix


def test_get_rotary_matrix():
    config = {
        "d_model": 128,
        "context_window": 16,
    }

    R = get_rotary_matrix(config["context_window"], config["d_model"])
    x = torch.randn(config["d_model"])
    y = torch.randn(config["d_model"])

    m = 3
    n = 13

    x_m = R[m, :, :] @ x
    x_n = R[n, :, :] @ y

    assert torch.isclose(x_m @ x_n, x @ R[n - m, :, :] @ y)


def test_rope_attention():
    config = {
        "batch_size": 10,
        "d_model": 512,
        "n_heads": 8,
        "context_window": 16,
    }
    layer = RoPEAttention(config)
    x = torch.randn((config["batch_size"], config["context_window"], config["d_model"]))

    q = layer.w_q(x)
    k = layer.w_k(x)
    v = layer.w_v(x)

    q_rotated = torch.zeros_like(x)
    k_rotated = torch.zeros_like(x)
    v_rotated = torch.zeros_like(x)

    for position in range(config["context_window"]):
        q_rotated[:, position, :] = torch.matmul(
            q[:, position, :], layer.R[position, :, :]
        )
        k_rotated[:, position, :] = torch.matmul(
            k[:, position, :], layer.R[position, :, :]
        )
        v_rotated[:, position, :] = torch.matmul(
            v[:, position, :], layer.R[position, :, :]
        )

    q_out = (torch.bmm(q.transpose(0, 1), layer.R)).transpose(0, 1)
    k_out = (torch.bmm(k.transpose(0, 1), layer.R)).transpose(0, 1)
    v_out = (torch.bmm(v.transpose(0, 1), layer.R)).transpose(0, 1)

    assert torch.allclose(q.transpose(0, 1)[0], q[:, 0, :])
    assert torch.allclose(q.transpose(0, 1)[0] @ layer.R[0], q[:, 0, :] @ layer.R[0])
    assert torch.allclose(q_rotated, q_out)


def test_rope_attention_small():
    config = {
        "batch_size": 1,
        "d_model": 2,
        "n_heads": 2,
        "context_window": 3,
    }

    layer = RoPEAttention(config)
    batch = torch.ones(
        (config["batch_size"], config["context_window"], config["d_model"])
    )
    output, attn_weights = layer(batch, return_attn_weights=True)

    m = 0
    x_q = batch[0, m]
    q = layer.R[m, :, :] @ layer.w_q(x_q)

    assert torch.allclose(layer.w_q(x_q), layer.w_q.weight @ x_q)
    assert torch.allclose(q, layer.R[m, :, :] @ layer.w_q.weight @ x_q)

    n = 2
    x_k = batch[0, n]
    k = layer.R[n, :, :] @ layer.w_k(x_k)

    assert torch.allclose(layer.w_k(x_k), layer.w_k.weight @ x_k)
    assert torch.allclose(k, layer.R[n, :, :] @ layer.w_k.weight @ x_k)

    assert q.T @ k == q @ k  # transpose is redundant
    assert torch.allclose(
        q @ k,
        x_k.T
        @ layer.w_k.weight.T
        @ layer.R[n, :, :].T
        @ layer.R[m, :, :]
        @ layer.w_q.weight
        @ x_q,
    )
    assert torch.allclose(
        q @ k,
        x_k.T @ layer.w_k.weight.T @ layer.R[n - m, :, :].T @ layer.w_q.weight @ x_q,
    )
