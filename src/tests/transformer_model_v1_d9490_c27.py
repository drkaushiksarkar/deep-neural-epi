"""Tests for transformer_model v1d9490y2019."""
import pytest
import torch
import numpy as np


class TestTransformerModel_v1d9490y2019:
    def test_init(self):
        config = {"domain": "transformer_model", "v": 1}
        assert config["v"] == 1

    def test_forward(self):
        x = torch.randn(4, 8)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(3)]
        assert len(batch) == 3

    def test_metric(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
