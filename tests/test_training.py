import pytest
import torch

from model import SimpleBrokenModel, SimpleModel, SimpleModel_RMS
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
        }

    def test_simple_broken(self):
        model = SimpleBrokenModel(self.MASTER_CONFIG)

        optimizer = torch.optim.Adam(model.parameters())

        loss_plot = train(model, optimizer, self.dataset, config=self.MASTER_CONFIG)

        assert loss_plot["train"].values[-1] <= 3.95
        assert loss_plot["val"].values[-1] <= 3.95

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
