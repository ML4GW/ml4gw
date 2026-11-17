import pytest

from ml4gw.nn.streaming import StreamWhitener
from ml4gw.transforms import SpectralDensity, Whiten


@pytest.fixture
def make_whitener():
    def make(psd_sample_rate=2048, **kwargs):
        defaults = {
            "kernel_length": 1,
            "psd_length": 8,
            "inference_sampling_rate": 128,
            "batch_size": 4,
        }
        return StreamWhitener(
            spectral_density=SpectralDensity(
                psd_sample_rate, fftlength=2, average="median", fast=True
            ),
            transform=Whiten(fduration=2, sample_rate=2048),
            **(defaults | kwargs),
        )

    return make
