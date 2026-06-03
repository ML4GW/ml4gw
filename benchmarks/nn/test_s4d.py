"""Benchmarks for the S4D sequence model."""

import pytest
import torch
from constants import NUM_CHANNELS

from ml4gw.nn.ssm.s4d import S4Model


@pytest.fixture(params=[512, 2048, 8192], ids=lambda n: f"L_{n}")
def seq_len(request):
    return request.param


def test_s4dmodel_forward(benchmark, batch_size, seq_len, device, maybe_sync):
    model = (
        S4Model(d_input=NUM_CHANNELS, d_output=1, d_model=256, n_layers=4)
        .to(device)
        .eval()
    )
    x = torch.randn(batch_size, NUM_CHANNELS, seq_len, device=device)
    with torch.no_grad():
        benchmark(maybe_sync(model), x)
