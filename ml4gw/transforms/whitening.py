from typing import Optional

import torch

from ml4gw import spectral
from ml4gw.transforms.transform import FittableSpectralTransform


class Whiten(torch.nn.Module):
    """
    Normalize the frequency content of timeseries
    data by a provided power spectral density, such
    that if the timeseries are sampled from the same
    distribution as the PSD the normalized power will
    be approximately unity across all frequency bins.
    The whitened timeseries will then also have
    0 mean and unit variance.

    In order to avoid edge effects due to filter settle-in,
    the provided PSDs will have their spectrum truncated
    such that their impulse response time in the time
    domain is `fduration` seconds, and `fduration / 2`
    seconds worth of data will be removed from each
    edge of the whitened timeseries.

    For more information, see the documentation to
    `ml4gw.spectral.whiten`.

    Args:
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. `fduration / 2` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        sample_rate:
            Rate at which timeseries data passed at call
            time is expected to be sampled
        highpass:
            Cutoff frequency to apply highpass filtering
            during whitening. If left as `None`, no highpass
            filtering will be performed.
    """

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
        """
        Whiten a batch of multichannel timeseries by a
        background power spectral density.

        Args:
            X:
                Batch of multichannel timeseries to whiten.
                Should have the shape (B, C, N), where
                B is the batch size, C is the number of
                channels, and N is the number of seconds
                in the timeseries times `self.sample_rate`.
            psd:
                Power spectral density used to whiten the
                provided timeseries. Can be either 1D, 2D,
                or 3D, with the last dimension representing
                power at each frequency value. All other
                dimensions must match their corresponding
                value in `X`, starting from the right.
                (e.g. if `psd.ndim == 2`, `psd.size(1)` should
                be equal to `X.size(1)`. If `psd.ndim == 3`,
                `psd.size(1)` and `psd.size(0)` should be equal
                to `X.size(1)` and `X.size(0)`, respectively.)
                For more information about what these different
                shapes for `psd` represent, consult the documentation
                for `ml4gw.spectral.whiten`.
        Returns:
            Whitened timeseries, with `fduration * sample_rate / 2`
                samples cropped from each edge. Output shape will then
                be (B, C, N - `fduration * sample_rate`).
        """

        return spectral.whiten(
            X,
            psd,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
        )


class FixedWhiten(FittableSpectralTransform):
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
        dtype: torch.dtype = torch.float64,
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
        *background: torch.Tensor,
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
            x = self.normalize_psd(
                x, self.sample_rate, num_freqs, fftlength, overlap
            )
            x = x.view(1, 1, -1)

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
