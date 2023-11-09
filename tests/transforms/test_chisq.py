import torch

from ml4gw.transforms import ChiSq, SpectralDensity


def test_chisq():
    scale = 10**(-19)
    background = scale * torch.randn(4, 2, 2048 * 32)
    strain = scale * torch.randn(4, 2, 2048 * 4)

    t = torch.arange(2048 * 4) / 2048
    freq = 10 + t**3
    amp = scale * (0.1 + t**3/64)
    signal = amp * torch.sin(2 * torch.pi * freq * t)
    signal = signal.view(1, 1, -1).repeat(4, 2, 1)
    injected = strain + signal

    spec = SpectralDensity(
        sample_rate=2048,
        fftlength=4,
        overlap=2,
        average="median"
    )
    psd = spec(background)

    transform = ChiSq(num_bins=8, fftlength=4, sample_rate=2048, highpass=10)
    chisq = transform(signal, injected, psd)
    print(chisq)
    raise ValueError
