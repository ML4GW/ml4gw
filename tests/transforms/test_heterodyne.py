import pytest
import torch

from ml4gw.transforms.heterodyne import Heterodyne


def test_time_output_shape():
    sample_rate = 2048
    kernel_length = 4
    batch = 5
    channels = 2
    chirp_mass = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.0)), torch.log(torch.tensor(2.5)), 5
        )
    )

    heterodyne = Heterodyne(
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        chirp_mass=chirp_mass,
        return_type="time",
    )

    X = torch.randn(batch, channels, int(sample_rate * kernel_length))
    out = heterodyne(X)
    assert out.shape == (
        batch,
        channels,
        5,
        int(sample_rate * kernel_length),
    )


def test_freq_output_shape():
    sample_rate = 2048
    kernel_length = 4
    batch = 5
    channels = 2
    chirp_mass = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.0)), torch.log(torch.tensor(2.5)), 5
        )
    )
    freqs = int(sample_rate * kernel_length) // 2 + 1

    heterodyne = Heterodyne(
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        chirp_mass=chirp_mass,
        return_type="freq",
    )

    X = torch.randn(batch, channels, int(sample_rate * kernel_length))
    out = heterodyne(X)
    assert out.shape == (batch, channels, 5, freqs)


def test_return_both():
    sample_rate = 2048
    kernel_length = 4
    batch = 5
    channels = 2
    chirp_mass = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.0)), torch.log(torch.tensor(2.5)), 5
        )
    )
    freqs = int(sample_rate * kernel_length) // 2 + 1

    heterodyne = Heterodyne(
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        chirp_mass=chirp_mass,
        return_type="both",
    )

    X = torch.randn(batch, channels, int(sample_rate * kernel_length))
    time, freq = heterodyne(X)
    assert time.shape == (
        batch,
        channels,
        5,
        int(sample_rate * kernel_length),
    )
    assert freq.shape == (batch, channels, 5, freqs)


def test_single_chirp_mass():
    sample_rate = 2048
    kernel_length = 4
    batch = 5
    channels = 2
    chirp_mass = torch.tensor([2.5])

    heterodyne = Heterodyne(
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        chirp_mass=chirp_mass,
        return_type="time",
    )

    X = torch.randn(batch, channels, int(sample_rate * kernel_length))
    out = heterodyne(X)
    assert out.shape[2] == 1
    assert out.shape == (
        batch,
        channels,
        1,
        int(sample_rate * kernel_length),
    )


def test_invalid_return_type_init():
    with pytest.raises(ValueError):
        Heterodyne(
            sample_rate=2048,
            kernel_length=4,
            chirp_mass=torch.tensor([2.5]),
            return_type="timeseries",
        )


def test_missing_chirp_mass_grid():
    with pytest.raises(ValueError):
        Heterodyne(
            sample_rate=2048,
            kernel_length=4,
            return_type="time",
        )
