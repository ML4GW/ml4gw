from typing import Tuple

import torch


def colored_gaussian_noise(
    shape: Tuple[int, ...], psd: torch.Tensor, sample_rate: float
):
    """
    Generate time-domain Gaussian noise colored by a specified PSD.

    Args:
        shape:
            Shape of noise tensor to generate.
            Last dimension corresponds to the time dimension.
        psd:
            Spectral density used to color noise
        sample_rate:
            Sampling rate of data

    Returns:
        torch.Tensor:
            Colored Gaussian noise
    """

    noise = torch.randn(shape)
    noise_fft = torch.fft.fft(noise)

    asd = torch.sqrt(psd)
    colored_fft = noise_fft * asd

    colored = torch.fft.ifft(colored_fft)
    colored /= torch.std(colored)

    return colored.real
