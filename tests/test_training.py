from copy import deepcopy

import pytest
import torch

from llama import Llama
from model import (
    RopeModel,
    RopeModelSwish,
    SimpleBrokenModel,
    SimpleModel,
    SimpleModel_RMS,
)
from train import train
from utils import get_batches, get_dataset


class TestTraining:
    @classmethod
    def setup_class(cls):
        cls.dataset, cls.vocab = get_dataset()
        cls.MASTER_CONFIG = {
            "vocab_size": len(cls.vocab),
            "batch_size": 32,
            "context_window": 16,
            "d_model": 128,
            "epochs": 1000,
            "log_interval": 10,
            "n_heads": 8,
        }

    def test_simple_broken(self):
        model = SimpleBrokenModel(self.MASTER_CONFIG)

        optimizer = torch.optim.Adam(model.parameters())

        loss_plot = train(model, optimizer, self.dataset, config=self.MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 3.95
        assert loss_plot["val"].values[-1] <= 3.98

    def test_simple_model(self):
        model = SimpleModel(self.MASTER_CONFIG)

        optimizer = torch.optim.Adam(model.parameters())

        loss_plot = train(model, optimizer, self.dataset, config=self.MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 2.60
        assert loss_plot["val"].values[-1] <= 2.60

    def test_simple_model_rms(self):
        model = SimpleModel_RMS(self.MASTER_CONFIG)

        optimizer = torch.optim.Adam(model.parameters())

        loss_plot = train(model, optimizer, self.dataset, config=self.MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 2.60
        assert loss_plot["val"].values[-1] <= 2.60

    def test_rope_model(self):
        model = RopeModel(self.MASTER_CONFIG)
        xs, ys = get_batches(
            self.dataset,
            "train",
            self.MASTER_CONFIG["batch_size"],
            self.MASTER_CONFIG["context_window"],
        )

        logits, loss = model(xs, ys)
        optimizer = torch.optim.Adam(model.parameters())
        loss_plot = train(model, optimizer, self.dataset, config=self.MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 2.2
        assert loss_plot["val"].values[-1] <= 2.2

    def test_rope_model_5000_epochs(self):
        MASTER_CONFIG = deepcopy(self.MASTER_CONFIG)
        MASTER_CONFIG.update(
            {
                "epochs": 5000,
                "log_interval": 10,
            }
        )
        model = RopeModel(MASTER_CONFIG)
        xs, ys = get_batches(
            self.dataset,
            "train",
            MASTER_CONFIG["batch_size"],
            MASTER_CONFIG["context_window"],
        )

        logits, loss = model(xs, ys)
        optimizer = torch.optim.Adam(model.parameters())
        loss_plot = train(model, optimizer, self.dataset, config=MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 1.95
        assert loss_plot["val"].values[-1] <= 2.0

    def test_rope_model_swish(self):
        MASTER_CONFIG = deepcopy(self.MASTER_CONFIG)
        MASTER_CONFIG.update(
            {
                "epochs": 5000,
                "log_interval": 10,
            }
        )
        model = RopeModelSwish(MASTER_CONFIG)
        xs, ys = get_batches(
            self.dataset,
            "train",
            MASTER_CONFIG["batch_size"],
            MASTER_CONFIG["context_window"],
        )

        logits, loss = model(xs, ys)
        optimizer = torch.optim.Adam(model.parameters())
        loss_plot = train(model, optimizer, self.dataset, config=MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 1.88
        assert loss_plot["val"].values[-1] <= 2.0

    @pytest.mark.parametrize(
        "epochs,train_loss,val_loss",
        [
            (5000, 1.65, 1.8),
            # ... add more sets of parameters as needed
        ],
    )
    def test_llama(self, epochs, train_loss, val_loss):
        MASTER_CONFIG = deepcopy(self.MASTER_CONFIG)
        MASTER_CONFIG.update(
            {
                "n_layers": 4,
                "epochs": epochs,
                "log_interval": 10,  # Keeping this constant, you can parameterize it too if needed.
            }
        )
        llama = Llama(MASTER_CONFIG)
        optimizer = torch.optim.Adam(llama.parameters())
        loss_plot = train(llama, optimizer, self.dataset, config=MASTER_CONFIG)
        assert loss_plot["train"].values[-1] <= train_loss
        assert loss_plot["val"].values[-1] <= val_loss
