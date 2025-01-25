import numpy as np
import pytest
import torch
from scipy.signal import butter, filtfilt

from ml4gw.transforms.iirfilter import IIRFilter


@pytest.fixture(params=[256, 512, 1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 6, 8])
def order(request):
    return request.param


def test_butterworth(sample_rate, order):
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone_freq = 50
    noise_amplitude = 0.5

    signal = np.sin(2 * np.pi * tone_freq * t)
    noise = noise_amplitude * np.random.normal(size=t.shape)
    combined_signal = signal + noise

    low_cutoff = 100
    high_cutoff = 20
    slice_length = int(0.1 * sample_rate)

    butterworth_low = IIRFilter(
        order,
        low_cutoff,
        btype="low",
        analog=False,
        fs=sample_rate,
    )
    butterworth_high = IIRFilter(
        order,
        high_cutoff,
        btype="high",
        analog=False,
        fs=sample_rate,
    )

    b, a = butter(
        order,
        low_cutoff,
        btype="low",
        analog=False,
        output="ba",
        fs=sample_rate,
    )
    scipy_filtered_data_low = filtfilt(b, a, combined_signal)[
        slice_length:-slice_length
    ]
    b, a = butter(
        order,
        high_cutoff,
        btype="high",
        analog=False,
        output="ba",
        fs=sample_rate,
    )
    scipy_filtered_data_high = filtfilt(b, a, combined_signal)[
        slice_length:-slice_length
    ]

    torch_filtered_data_low = butterworth_low(
        torch.tensor(combined_signal).repeat(10, 1)
    )[:, slice_length:-slice_length].numpy()
    torch_filtered_data_high = butterworth_high(
        torch.tensor(combined_signal).repeat(10, 1)
    )[:, slice_length:-slice_length].numpy()

    # test batch processing
    for i in range(9):
        assert np.allclose(
            torch_filtered_data_low[0],
            torch_filtered_data_low[i + 1],
            atol=float(np.finfo(float).eps),
        )
        assert np.allclose(
            torch_filtered_data_high[0],
            torch_filtered_data_high[i + 1],
            atol=float(np.finfo(float).eps),
        )

    assert np.allclose(
        scipy_filtered_data_low,
        torch_filtered_data_low[0],
        atol=1e-1,
    )
    assert np.allclose(
        scipy_filtered_data_high,
        torch_filtered_data_high[0],
        atol=1e-1,
    )
