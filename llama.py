from collections import OrderedDict

import torch.nn as nn
from torch.nn import functional as F

from model import RMSNorm, RoPEAttention_wMask, SwiGLU


# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config["context_window"], config["d_model"]))

        self.attention = RoPEAttention_wMask(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            SwiGLU(config["d_model"]),
        )

    def forward(self, x):
        x = self.rms(x)  # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x)  # rms pre-normalization
        x = x + self.feedforward(x)
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        self.llama_blocks = nn.Sequential(
            OrderedDict(
                [(f"llama_{i}", LlamaBlock(config)) for i in range(config["n_layers"])]
            )
        )

        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            SwiGLU(config["d_model"]),
            nn.Linear(config["d_model"], config["vocab_size"]),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss
