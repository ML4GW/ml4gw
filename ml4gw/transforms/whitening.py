from typing import Optional, Union

import numpy as np
import torch

from ml4gw import spectral
from ml4gw.transforms.transform import FittableTransform


class Whiten(torch.nn.Module):
    def __init__(
        self,
        fduration: float,
        sample_rate: float,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.fduration = fduration
        self.sample_rate = sample_rate
        self.highpass = highpass

        # register a window up front to signify our
        # fduration at inference time
        size = int(fduration * sample_rate)
        window = torch.hann_window(size, dtype=torch.float64)
        self.register_buffer("window", window)

    def forward(self, X: torch.Tensor, psd: torch.Tensor) -> torch.Tensor:
        return spectral.whiten(
            X,
            psd,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
        )


class FixedWhiten(FittableTransform):
    def __init__(
        self,
        num_channels: float,
        kernel_length: float,
        sample_rate: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length

        N = int(kernel_length * sample_rate)
        num_freqs = N // 2 + 1
        psd = torch.zeros((num_channels, num_freqs), dtype=dtype)
        self.register_buffer("psd", psd)

        # save this as a parameter since it's decided at fit time
        fduration = torch.zeros((1,))
        self.register_buffer("fduration", fduration)

    def fit(
        self,
        fduration: float,
        *background: Union[torch.Tensor, np.ndarray],
        fftlength: Optional[float] = None,
        highpass: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> None:
        if len(background) != self.num_channels:
            raise ValueError(
                "Expected to fit whitening transform on {} background "
                "timeseries, but was passed {}".format(
                    self.num_channels, len(background)
                )
            )

        num_freqs = self.psd.size(-1)
        psds = []
        for x in background:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

            if fftlength is not None:
                nperseg = int(fftlength * self.sample_rate)

                overlap = overlap or fftlength / 2
                nstride = nperseg - int(overlap * self.sample_rate)

                window = torch.hann_window(nperseg, dtype=torch.float64)
                scale = 1.0 / (self.sample_rate * (window**2).sum())
                x = spectral.spectral_density(
                    x,
                    nperseg=nperseg,
                    nstride=nstride,
                    window=window,
                    scale=scale,
                )

            x = x.view(1, 1, -1)
            if len(x) != num_freqs:
                x = torch.nn.functional.interpolate(x, size=(num_freqs,))

            psd = spectral.truncate_inverse_power_spectrum(
                x, fduration, self.sample_rate, highpass
            )
            psds.append(psd[0, 0])
        psd = torch.stack(psds)

        fduration = torch.Tensor([fduration])
        self.build(psd=psd, fduration=fduration)

    def forward(self, X):
        expected_dim = int(self.kernel_length * self.sample_rate)
        if X.size(-1) != expected_dim:
            raise ValueError(
                "Whitening transform expected a kernel length "
                "of {}s, but was passed data of length {}s".format(
                    self.kernel_length, X.size(-1) / self.sample_rate
                )
            )

        pad = int(self.fduration.item() * self.sample_rate / 2)
        return spectral.normalize_by_psd(X, self.psd, self.sample_rate, pad)
