"""Benchmarks for SingleQTransform and QScan."""

import pytest
import torch
from constants import KERNEL_LEN, NUM_CHANNELS

from ml4gw.transforms import QScan, SingleQTransform

SPECTROGRAM_SHAPE = (64, 64)


@pytest.fixture(
    params=[(1024, 12.0), (4096, 12.0), (4096, 64.0)],
    ids=lambda x: f"sr_{x[0]}_q_{int(x[1])}",
)
def qtransform_config(request):
    return request.param


def test_single_qtransform_compute_qtiles(
    benchmark, batch_size, qtransform_config, device
):
    sample_rate, q = qtransform_config
    qtransform = SingleQTransform(
        duration=KERNEL_LEN,
        sample_rate=sample_rate,
        spectrogram_shape=SPECTROGRAM_SHAPE,
        q=q,
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, sample_rate, device=device)
    benchmark(qtransform.compute_qtiles, x)


def test_single_qtransform_forward(
    benchmark, batch_size, qtransform_config, device
):
    sample_rate, q = qtransform_config
    qtransform = SingleQTransform(
        duration=KERNEL_LEN,
        sample_rate=sample_rate,
        spectrogram_shape=SPECTROGRAM_SHAPE,
        q=q,
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, sample_rate, device=device)
    benchmark(qtransform, x)


@pytest.fixture(params=[1024, 4096], ids=lambda x: f"sr_{x}")
def qscan_sample_rate(request):
    return request.param


def test_qscan_forward(benchmark, batch_size, qscan_sample_rate, device):
    qscan = QScan(
        duration=KERNEL_LEN,
        sample_rate=qscan_sample_rate,
        spectrogram_shape=SPECTROGRAM_SHAPE,
        qrange=[1, 60],
    ).to(device)
    x = torch.randn(batch_size, NUM_CHANNELS, qscan_sample_rate, device=device)
    benchmark(qscan, x)
