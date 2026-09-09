import torch

from .. import spectral
from ..types import (
    FrequencySeries1d,
    FrequencySeries1to3d,
    TimeSeries1d,
    TimeSeries3d,
)
from .transform import FittableSpectralTransform


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
    domain is ``fduration`` seconds, and ``fduration / 2``
    seconds worth of data will be removed from each
    edge of the whitened timeseries.

    For more information, see the documentation for
    :meth:`~ml4gw.spectral.whiten`.

    Args:
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. ``fduration / 2`` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        sample_rate:
            Rate at which timeseries data passed at call
            time is expected to be sampled
        highpass:
            Cutoff frequency to apply highpass filtering
            during whitening. If left as ``None``, no highpass
            filtering will be performed.
        lowpass:
            Cutoff frequency to apply lowpass filtering
            during whitening. If left as ``None``, no lowpass
            filtering will be performed.
    """

    def __init__(
        self,
        fduration: float,
        sample_rate: float,
        highpass: float | None = None,
        lowpass: float | None = None,
    ) -> None:
        super().__init__()
        self.fduration = fduration
        self.sample_rate = sample_rate
        self.highpass = highpass
        self.lowpass = lowpass

        # register a window up front to signify our
        # fduration at inference time
        size = int(fduration * sample_rate)
        window = torch.hann_window(size, dtype=torch.float64)
        self.register_buffer("window", window)

    def forward(
        self,
        X: TimeSeries3d,
        psd: FrequencySeries1to3d,
        crop: bool = True,
    ) -> TimeSeries3d:
        """
        Whiten a batch of multichannel timeseries by a
        background power spectral density.

        Args:
            X:
                Batch of multichannel timeseries to whiten.
                Should have the shape ``(B, C, N)``, where
                ``B`` is the batch size, ``C`` is the number of
                channels, and ``N`` is the number of seconds
                in the timeseries times ``self.sample_rate``.
            psd:
                Power spectral density used to whiten the
                provided timeseries. Can be either 1D, 2D,
                or 3D, with the last dimension representing
                power at each frequency value. All other
                dimensions must match their corresponding
                value in ``X``, starting from the right.
                (e.g. if ``psd.ndim == 2``, ``psd.size(1)`` should
                be equal to ``X.size(1)``. If ``psd.ndim == 3``,
                ``psd.size(1)`` and ``psd.size(0)`` should be equal
                to ``X.size(1)`` and ``X.size(0)``, respectively.)
                For more information about what these different
                shapes for ``psd`` represent, consult the documentation
                for :meth:`~ml4gw.spectral.whiten`.
            crop:
                If ``True``, crop ``fduration / 2`` seconds of data
                from both sides of the time dimension to remove the
                corruption from the filter. If ``False``, return the
                full timeseries.
        Returns:
            Whitened timeseries, with ``fduration * sample_rate / 2``
                samples cropped from each edge. Output shape will then
                be ``(B, C, N - fduration * sample_rate)``.
        """

        return spectral.whiten(
            X,
            psd,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
            lowpass=self.lowpass,
            crop=crop,
        )


class FixedWhiten(FittableSpectralTransform):
    """
    Transform that whitens timeseries by a fixed
    power spectral density that's determined by
    calling the ``.fit`` method.

    Args:
        num_channels:
            Number of channels to whiten
        kernel_length:
            Expected length of tensors to whiten
            in seconds. Determines the number of
            frequency bins in the fit PSD.
        sample_rate:
            Rate at which timeseries will be sampled, in Hz
        crop:
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
        *background: TimeSeries1d | FrequencySeries1d,
        fftlength: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        overlap: float | None = None,
    ) -> None:
        """
        Compute the PSD of channel-wise background to
        use to whiten timeseries at call time. PSDs will
        be resampled to have
        ``self.kernel_length * self.sample_rate // 2 + 1``
        frequency bins.

        Args:
            fduration:
                Desired length of the impulse response
                of the whitening filter, in seconds.
                Fit PSDs will have their spectrum truncated
                to approximate this response time.
                A longer ``fduration`` will be able to
                handle narrower spikes in frequency, but
                at the expense of longer filter settle-in
                time. As such ``fduration / 2`` seconds of data
                will be removed from each edge of whitened
                timeseries.
            *background:
                1D arrays capturing the signal to be used to
                whiten each channel at call time. If ``fftlength``
                is left as ``None``, it will be assumed that these
                already represent frequency-domain data that will
                be possibly resampled and truncated to whiten
                timeseries at call time. Otherwise, it will be
                assumed that these represent time-domain data that
                will be converted to the frequency domain via
                Welch's method using the specified ``fftlength``
                and ``overlap``, with a Hann window used to window
                the FFT frames by default. Should have the same
                number of args as ``self.num_channels``.
            fftlength:
                Length of frames used to convert time-domain
                data to the frequency-domain via Welch's method.
                If left as ``None``, it will be assumed that the
                background arrays passed already represent frequency-
                domain data and don't require any conversion.
            highpass:
                Cutoff frequency, in Hz, used for highpass filtering
                with the fit whitening filter. This is achieved by
                setting the frequency response of the fit PSDs
                in the frequency bins below this value to 0.
                If left as ``None``, the fit filter won't have any
                highpass filtering properties.
            lowpass:
                Cutoff frequency, in Hz, used for lowpass filtering
                with the fit whitening filter. This is achieved by
                setting the frequency response of the fit PSDs
                in the frequency bins above this value to 0.
                If left as ``None``, the fit filter won't have any
                lowpass filtering properties.
            overlap:
                Overlap between FFT frames used to convert
                time-domain data to the frequency domain via
                Welch's method. If ``fftlength`` is ``None``, this
                is ignored. Otherwise, if left as ``None``, it will
                be set to half of ``fftlength`` by default.
        """
        if len(background) != self.num_channels:
            raise ValueError(
                f"Expected to fit whitening transform on {self.num_channels} "
                f"background timeseries, but was passed {len(background)}"
            )

        num_freqs = self.psd.size(-1)
        psds = []
        for x in background:
            x = self.normalize_psd(
                x, self.sample_rate, num_freqs, fftlength, overlap
            )
            x = x.view(1, 1, -1)

            psd = spectral.truncate_inverse_power_spectrum(
                x, fduration, self.sample_rate, highpass, lowpass
            )
            psds.append(psd[0, 0])
        psd = torch.stack(psds)

        fduration = torch.Tensor([fduration])
        self.build(psd=psd, fduration=fduration)

    def forward(self, X: TimeSeries3d, crop: bool = True) -> TimeSeries3d:
        """
        Whiten the input timeseries tensor using the
        PSD fit by the ``.fit`` method, which must be
        called **before** the first call to ``.forward``.

        Args:
            X:
                Batch of multichannel timeseries to whiten.
                Should have the shape ``(B, C, N)``, where
                ``B`` is the batch size, ``C`` is the number of
                channels, and ``N`` is the number of seconds
                in the timeseries times ``self.sample_rate``.
            crop:
                If ``True``, crop ``fduration / 2`` seconds of data
                from both sides of the time dimension to remove the
                corruption from the filter. If ``False``, return the
                full timeseries.
        """
        expected_dim = int(self.kernel_length * self.sample_rate)
        if X.size(-1) != expected_dim:
            raise ValueError(
                "Whitening transform expected a kernel length "
                f"of {self.kernel_length}s, but was passed data "
                f"of length {X.size(-1) / self.sample_rate}s"
            )

        pad = int(self.fduration.item() * self.sample_rate / 2)
        return spectral.normalize_by_psd(
            X, self.psd, self.sample_rate, pad, crop
        )


class MinimumPhaseWhiten(FittableSpectralTransform):
    """Whiten timeseries with a causal minimum-phase FIR filter.

    Unlike :class:`Whiten`, this transform never uses future samples to
    compute an output. The filter is fit once from a background PSD and then
    applied with left padding, which represents zero-valued history. Callers
    processing consecutive chunks should prepend real input history and
    discard the corresponding warm-up outputs. The transform does not remove
    a running mean, since doing so would introduce a future-sample dependency.

    Args:
        num_channels:
            Number of channels to whiten.
        kernel_length:
            Length of the whitening filter in seconds.
        sample_rate:
            Sampling rate of input timeseries in Hz.
        dtype:
            Datatype used to store the fitted filter.

    Shape:
        - Input: ``(B, C, T)``
        - Output: ``(B, C, T)``
    """

    def __init__(
        self,
        num_channels: int,
        kernel_length: float,
        sample_rate: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.kernel_length = kernel_length
        self.sample_rate = sample_rate

        size = int(kernel_length * sample_rate)
        if size < 2:
            raise ValueError(
                "Whitening filter must contain at least two samples"
            )
        kernel = torch.zeros((num_channels, 1, size), dtype=dtype)
        self.register_buffer("kernel", kernel)

    def fit(
        self,
        *background: TimeSeries1d | FrequencySeries1d,
        fftlength: float | None = None,
        overlap: float | None = None,
    ) -> None:
        """Fit a minimum-phase whitening filter to background data.

        Args:
            *background:
                One time- or frequency-domain tensor per channel. Inputs are
                interpreted as one-sided PSDs when ``fftlength`` is ``None``;
                otherwise Welch PSDs are estimated from the timeseries.
            fftlength:
                Length in seconds of Welch frames used for time-domain input.
            overlap:
                Overlap in seconds between Welch frames. Defaults to half of
                ``fftlength``.
        """
        if len(background) != self.num_channels:
            raise ValueError(
                f"Expected to fit whitening transform on {self.num_channels} "
                f"background timeseries, but was passed {len(background)}"
            )

        num_freqs = self.kernel.size(-1) // 2 + 1
        psds = [
            self.normalize_psd(
                x,
                self.sample_rate,
                num_freqs,
                fftlength,
                overlap,
            )
            for x in background
        ]
        psd = torch.stack(psds)
        kernel = spectral.minimum_phase_whitening_filter(
            psd, n_fft=self.kernel.size(-1)
        )

        # ML4GW PSDs are one-sided, so account for folded negative-frequency
        # power and retain the unit-variance convention used by ``Whiten``.
        kernel = kernel * (2 / self.sample_rate) ** 0.5
        self.build(kernel=kernel[:, None])

    def forward(self, X: TimeSeries3d) -> TimeSeries3d:
        """Apply the fitted causal filter using zero-valued initial history."""
        if X.ndim != 3 or X.size(1) != self.num_channels:
            raise ValueError(
                "Whitening transform expected input with shape "
                f"(batch, {self.num_channels}, time), but found {X.shape}"
            )

        kernel = self.kernel.to(X)
        X = torch.nn.functional.pad(X, (kernel.size(-1) - 1, 0))
        return torch.nn.functional.conv1d(
            X,
            torch.flip(kernel, dims=(-1,)),
            groups=self.num_channels,
        )
