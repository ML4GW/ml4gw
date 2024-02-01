from typing import Tuple

import torch


def colored_gaussian_noise(shape: Tuple[int, int, int], psd: torch.Tensor):
    """
    Generate time-domain Gaussian noise colored by a specified PSD.

    Args:
        shape:
            3D shape of noise tensor to generate.
            First dimension corresponds to `batch_size`,
            Second dimension corresponds to the number of channels,
            Last dimension corresponds to the time dimension.
        psd:
            Spectral density used to color noise
        sample_rate:
            Sampling rate of data

    Returns:
        torch.Tensor:
            Colored Gaussian noise
    """

    if len(shape) != 3:
        raise ValueError("Shape must have 3 dimensions")

    X = torch.randn(shape)
    # possibly interpolate our PSD to match the number
    # of frequency bins we expect to get from X
    N = X.size(-1)
    num_freqs = N // 2 + 1

    # normalize the number of expected dimensions in the PSD
    while psd.ndim < 3:
        psd = psd[None]

    if psd.size(-1) != num_freqs:
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs), mode="linear"
        )

    X_fft = torch.fft.rfft(X, norm="forward", dim=-1)
    X_fft *= psd**0.5
    X_fft = torch.fft.irfft(X_fft, norm="forward", dim=-1)
    X_fft /= torch.std(X_fft, dim=-1, keepdim=True)
    return X_fft
