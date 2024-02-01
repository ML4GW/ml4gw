import pytest
import torch

from ml4gw.simulation.noise import colored_gaussian_noise


@pytest.fixture
def sample_rate():
    return 2048


@pytest.fixture
def duration():
    return 10


def test_colored_gaussian_noise(sample_rate, duration):
    # test already white psd produces white noise
    shape = (10, 2, int(sample_rate * duration))
    n_freqs = int(sample_rate * duration) // 2 + 1
    psd = torch.ones(n_freqs)

    noise = colored_gaussian_noise(shape, psd)
    assert noise.shape == shape

    # test that the noise is white
    means = noise.mean(axis=-1)
    target = torch.zeros_like(means)
    torch.testing.assert_close(means, target, rtol=0, atol=0.02)

    stds = noise.std(axis=-1)
    target = torch.ones_like(stds)
    torch.testing.assert_close(stds, target, rtol=0.015, atol=0.0)
