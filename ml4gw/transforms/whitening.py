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
    """
    Transform that whitens timeseries by a fixed
    power spectral density that's determined by
    calling the `.fit` method.

    Args:
        num_channels:
            Number of channels to whiten
        kernel_length:
            Expected length of tensors to whiten
            in seconds. Determines the number of
            frequency bins in the fit PSD.
        sample_rate:
            Rate at which timeseries will be sampled, in Hz
        dtype:
            Datatype with which background PSD will be stored
    """

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
        """
        Compute the PSD of channel-wise background to
        use to whiten timeseries at call time. PSDs will
        be resampled to have
        `self.kernel_length * self.sample_rate // 2 + 1`
        frequency bins.

        Args:
            fduration:
                Desired length of the impulse response
                of the whitening filter, in seconds.
                Fit PSDs will have their spectrum truncated
                to approximate this response time.
                A longer `fduration` will be able to
                handle narrower spikes in frequency, but
                at the expense of longer filter settle-in
                time. As such `fduration / 2` seconds of data
                will be removed from each edge of whitened
                timeseries.
            *background:
                1D arrays capturing the signal to be used to
                whiten each channel at call time. If `fftlength`
                is left as `None`, it will be assumed that these
                already represent frequency-domain data that will
                be possibly resampled and truncated to whiten
                timeseries at call time. Otherwise, it will be
                assumed that these represent time-domain data that
                will be converted to the frequency domain via
                Welch's method using the specified `fftlength`
                and `overlap`, with a Hann window used to window
                the FFT frames by default. Should have the same
                number of args as `self.num_channels`.
            fftlength:
                Length of frames used to convert time-domain
                data to the frequency-domain via Welch's method.
                If left as `None`, it will be assumed that the
                background arrays passed already represent frequency-
                domain data and don't require any conversion.
            highpass:
                Cutoff frequency, in Hz, used for highpass filtering
                with the fit whitening filter. This is achieved by
                setting the frequency response of the fit PSDs
                in the frequency bins below this value to 0.
                If left as `None`, the fit filter won't have any
                highpass filtering properties.
            overlap:
                Overlap between FFT frames used to convert
                time-domain data to the frequency domain via
                Welch's method. If `fftlength` is `None`, this
                is ignored. Otherwise, if left as `None`, it will
                be set to half of `fftlength` by default.
        """
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

            # if we specified an FFT length, convert
            # the (assumed) time-domain data to the
            # frequency domain
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

            # add two dummy dimensions in case we need to inerpolate
            # the frequency dimension, since `interpolate` expects
            # a (batch, channel, spatial) formatted tensor as input
            x = x.view(1, 1, -1)
            if x.size(-1) != num_freqs:
                x = torch.nn.functional.interpolate(x, size=(num_freqs,))

            psd = spectral.truncate_inverse_power_spectrum(
                x, fduration, self.sample_rate, highpass
            )
            psds.append(psd[0, 0])
        psd = torch.stack(psds)

        fduration = torch.Tensor([fduration])
        self.build(psd=psd, fduration=fduration)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Whiten the input timeseries tensor using the
        PSD fit by the `.fit` method, which must be
        called _before_ the first call to `.forward`.
        """
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
