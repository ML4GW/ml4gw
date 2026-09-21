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
    data by a provided amplitude spectral density, such
    that if the timeseries are sampled from the same
    distribution as the ASD the normalized power will
    be approximately unity across all frequency bins.
    The whitened timeseries will then also have
    0 mean and unit variance.

    In order to avoid edge effects due to filter settle-in,
    the provided ASDs will have their spectrum truncated
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
        window = torch.hann_window(size, dtype=torch.float32)
        self.register_buffer("window", window)

    def forward(
        self,
        X: TimeSeries3d,
        asd: FrequencySeries1to3d,
        crop: bool = True,
        highpass: float | None = None,
        lowpass: float | None = None,
    ) -> TimeSeries3d:
        """
        Whiten a batch of multichannel timeseries by a
        background amplitude spectral density.

        Args:
            X:
                Batch of multichannel timeseries to whiten.
                Should have the shape ``(B, C, N)``, where
                ``B`` is the batch size, ``C`` is the number of
                channels, and ``N`` is the number of seconds
                in the timeseries times ``self.sample_rate``.
            asd:
                Amplitude spectral density used to whiten the
                provided timeseries. Can be either 1D, 2D,
                or 3D, with the last dimension representing
                amplitude at each frequency value. All other
                dimensions must match their corresponding
                value in ``X``, starting from the right.
                (e.g. if ``asd.ndim == 2``, ``asd.size(1)`` should
                be equal to ``X.size(1)``. If ``asd.ndim == 3``,
                ``asd.size(1)`` and ``asd.size(0)`` should be equal
                to ``X.size(1)`` and ``X.size(0)``, respectively.)
                For more information about what these different
                shapes for ``asd`` represent, consult the documentation
                for :meth:`~ml4gw.spectral.whiten`.
            crop:
                If ``True``, crop ``fduration / 2`` seconds of data
                from both sides of the time dimension to remove the
                corruption from the filter. If ``False``, return the
                full timeseries.
            highpass:
                Override the highpass cutoff frequency for this call.
                If ``None``, falls back to the value set at
                initialization.
            lowpass:
                Override the lowpass cutoff frequency for this call.
                If ``None``, falls back to the value set at
                initialization.
        Returns:
            Whitened timeseries, with ``fduration * sample_rate / 2``
                samples cropped from each edge. Output shape will then
                be ``(B, C, N - fduration * sample_rate)``.
        """
        highpass = highpass if highpass is not None else self.highpass
        lowpass = lowpass if lowpass is not None else self.lowpass

        return spectral.whiten(
            X,
            asd,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=highpass,
            lowpass=lowpass,
            crop=crop,
        )


class FixedWhiten(FittableSpectralTransform):
    """
    Transform that whitens timeseries by a fixed
    amplitude spectral density that's determined by
    calling the ``.fit`` method.

    Args:
        num_channels:
            Number of channels to whiten
        kernel_length:
            Expected length of tensors to whiten
            in seconds. Determines the number of
            frequency bins in the fit ASD.
        sample_rate:
            Rate at which timeseries will be sampled, in Hz
        crop:
            Datatype with which background ASD will be stored
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
        asd = torch.zeros((num_channels, num_freqs), dtype=dtype)
        self.register_buffer("asd", asd)

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
        Compute the ASD of channel-wise background to
        use to whiten timeseries at call time. ASDs will
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

        num_freqs = self.asd.size(-1)
        asds = []
        for x in background:
            x = self.normalize_asd(
                x, self.sample_rate, num_freqs, fftlength, overlap
            )
            x = x.view(1, 1, -1)

            asd = spectral.truncate_inverse_amplitude_spectrum(
                x, fduration, self.sample_rate, highpass, lowpass
            )
            asds.append(asd[0, 0])
        asd = torch.stack(asds)

        fduration = torch.Tensor([fduration])
        self.build(asd=asd, fduration=fduration)

    def forward(self, X: TimeSeries3d, crop: bool = True) -> TimeSeries3d:
        """
        Whiten the input timeseries tensor using the
        ASD fit by the ``.fit`` method, which must be
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
        return spectral.normalize_by_asd(
            X, self.asd, self.sample_rate, pad, crop
        )


def _validate_minimum_phase_input(
    X: TimeSeries3d,
    asd: FrequencySeries1to3d | None = None,
    num_channels: int | None = None,
) -> None:
    if X.ndim != 3:
        raise ValueError(
            "Whitening transform expected input with shape "
            f"(batch, channels, time), but found {X.shape}"
        )
    if num_channels is not None and X.size(1) != num_channels:
        raise ValueError(
            "Whitening transform expected input with shape "
            f"(batch, {num_channels}, time), but found {X.shape}"
        )
    if asd is None:
        return
    if asd.ndim < 1 or asd.ndim > 3:
        raise ValueError("ASD must have one, two, or three dimensions")
    if asd.ndim >= 2 and asd.size(-2) != X.size(1):
        raise ValueError(
            f"ASD has {asd.size(-2)} channels, but input has {X.size(1)}"
        )
    if asd.ndim == 3 and asd.size(0) != X.size(0):
        raise ValueError(
            f"ASD has batch size {asd.size(0)}, but input has {X.size(0)}"
        )


def _minimum_phase_kernel(
    asd: FrequencySeries1to3d,
    size: int,
    sample_rate: float,
    highpass: float | None = None,
    lowpass: float | None = None,
    validate: bool = False,
) -> torch.Tensor:
    """Build a minimum-phase kernel from the standard truncated ASD.

    The ASD is first processed by
    :func:`~ml4gw.spectral.truncate_inverse_amplitude_spectrum`, including any
    requested highpass or lowpass response, then spectrally factorized into a
    causal FIR filter.
    """
    if asd.size(-1) < 2:
        raise ValueError("ASD must contain at least two frequency bins")
    if not torch.is_floating_point(asd):
        raise ValueError("ASD must be a floating-point tensor")

    n_fft = 2 * (asd.size(-1) - 1)

    # Construct the response at no less than twice the final FIR duration.
    # This preserves resolved spectral structure during factorization instead
    # of first downsampling the ASD to the final number of filter taps.
    if n_fft < 2 * size:
        shape = asd.shape[:-1]
        asd = torch.nn.functional.interpolate(
            asd.reshape(-1, 1, asd.size(-1)),
            size=size + 1,
            mode="linear",
            align_corners=True,
        ).reshape(*shape, size + 1)

    input_ndim = asd.ndim
    while asd.ndim < 3:
        asd = asd[None]
    asd = spectral.truncate_inverse_amplitude_spectrum(
        asd,
        size / sample_rate,
        sample_rate,
        highpass,
        lowpass,
    )
    kernel = spectral.minimum_phase_whitening_filter(asd, validate=validate)[
        ..., :size
    ]
    while kernel.ndim > input_ndim:
        kernel = kernel[0]

    # ``truncate_inverse_amplitude_spectrum`` accounts for the one-sided
    # convention. Retain the unit-variance normalization used by ``Whiten``.
    return kernel / sample_rate**0.5


def _minimum_phase_filter(
    X: TimeSeries3d,
    kernel: torch.Tensor,
    crop: bool,
) -> TimeSeries3d:
    """Apply a causal FIR as a linear convolution in the frequency domain."""
    size = kernel.size(-1)
    input_size = X.size(-1)
    if crop and input_size <= size:
        raise ValueError(
            f"Not enough timeseries samples {X.size(-1)} for number of "
            f"cropped samples {size}"
        )

    dtype = torch.promote_types(X.dtype, kernel.dtype)
    X = X.to(dtype=dtype)
    kernel = kernel.to(device=X.device, dtype=dtype)
    while kernel.ndim < X.ndim:
        kernel = kernel.unsqueeze(0)

    convolution_size = input_size + size - 1
    n_fft = 1 << (convolution_size - 1).bit_length()
    X_tilde = torch.fft.rfft(X, n=n_fft, dim=-1)
    kernel_tilde = torch.fft.rfft(kernel, n=n_fft, dim=-1)
    X = torch.fft.irfft(X_tilde * kernel_tilde, n=n_fft, dim=-1)
    X = X[..., :input_size]
    if crop:
        X = X[..., size:]
    return X.float()


class MinimumPhaseWhiten(torch.nn.Module):
    """Whiten timeseries with a dynamic causal minimum-phase FIR filter.

    A whitening kernel is constructed from the ASD provided at call time. It
    uses only current and previous samples. By default, the initial warm-up
    samples that depend on zero-valued history are removed from the output,
    along with one additional sample to match :class:`Whiten` output lengths.
    Unlike :class:`Whiten`, it does not subtract the mean of the complete
    input segment because that operation depends on future samples.

    Args:
        fduration:
            Length of the causal whitening filter in seconds.
        sample_rate:
            Sampling rate of input timeseries in Hz.
        highpass:
            Optional highpass frequency in Hz. The response is constructed by
            :func:`~ml4gw.spectral.truncate_inverse_amplitude_spectrum`, as in
            :class:`Whiten`.
        lowpass:
            Optional lowpass frequency in Hz. The response is constructed by
            :func:`~ml4gw.spectral.truncate_inverse_amplitude_spectrum`, as in
            :class:`Whiten`.
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
        self.size = int(fduration * sample_rate)
        if self.size < 2:
            raise ValueError(
                "Whitening filter must contain at least two samples"
            )

    def forward(
        self,
        X: TimeSeries3d,
        asd: FrequencySeries1to3d,
        crop: bool = True,
    ) -> TimeSeries3d:
        """Whiten input using an ASD supplied at call time.

        Args:
            X:
                Batch of multichannel timeseries with shape ``(B, C, T)``.
            asd:
                One-sided ASD with shape ``(F,)``, ``(C, F)``, or
                ``(B, C, F)``. Unscaled physical ASDs may require double
                precision to avoid underflow.
            crop:
                If ``True``, remove ``fduration * sample_rate`` samples from
                the left edge. If ``False``, return the full timeseries,
                including samples computed from zero-valued initial history.

        Returns:
            A ``torch.float32`` tensor with shape ``(B, C, T - L)`` when
            cropped, where ``L = int(fduration * sample_rate)``, or
            ``(B, C, T)`` otherwise.
        """
        _validate_minimum_phase_input(X, asd)
        kernel = _minimum_phase_kernel(
            asd,
            self.size,
            self.sample_rate,
            self.highpass,
            self.lowpass,
        )
        return _minimum_phase_filter(X, kernel, crop)


class FixedMinimumPhaseWhiten(FittableSpectralTransform):
    """Whiten timeseries with a fitted causal minimum-phase FIR filter.

    This transform does not subtract the mean of the complete input segment
    because that operation depends on future samples.

    Args:
        num_channels:
            Number of channels to whiten.
        fduration:
            Length of the causal whitening filter in seconds.
        sample_rate:
            Sampling rate of input timeseries in Hz.
        dtype:
            Datatype used to store the fitted filter.
    """

    def __init__(
        self,
        num_channels: int,
        fduration: float,
        sample_rate: float,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.fduration = fduration
        self.sample_rate = sample_rate

        size = int(fduration * sample_rate)
        if size < 2:
            raise ValueError(
                "Whitening filter must contain at least two samples"
            )
        kernel = torch.zeros((num_channels, size), dtype=dtype)
        self.register_buffer("kernel", kernel)

    def fit(
        self,
        *background: TimeSeries1d | FrequencySeries1d,
        fftlength: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        overlap: float | None = None,
    ) -> None:
        """Fit a minimum-phase whitening filter to background data.

        Args:
            *background:
                One time- or frequency-domain tensor per channel. Inputs are
                interpreted as one-sided ASDs when ``fftlength`` is ``None``;
                otherwise Welch ASDs are estimated from the timeseries.
                Unscaled physical ASDs may require double precision to avoid
                underflow.
            fftlength:
                Length in seconds of Welch frames used for time-domain input.
            highpass:
                Optional highpass frequency in Hz. Applied by
                :func:`~ml4gw.spectral.truncate_inverse_amplitude_spectrum`
                before minimum-phase factorization.
            lowpass:
                Optional lowpass frequency in Hz. Applied by
                :func:`~ml4gw.spectral.truncate_inverse_amplitude_spectrum`
                before minimum-phase factorization.
            overlap:
                Overlap in seconds between Welch frames. Defaults to half of
                ``fftlength``.
        """
        if len(background) != self.num_channels:
            raise ValueError(
                f"Expected to fit whitening transform on {self.num_channels} "
                f"background timeseries, but was passed {len(background)}"
            )

        asds = []
        for x in background:
            if fftlength is None:
                num_freqs = x.size(-1)
            else:
                num_freqs = int(fftlength * self.sample_rate) // 2 + 1
            asds.append(
                self.normalize_asd(
                    x,
                    self.sample_rate,
                    num_freqs,
                    fftlength,
                    overlap,
                )
            )
        asd = torch.stack(asds)
        kernel = _minimum_phase_kernel(
            asd,
            self.kernel.size(-1),
            self.sample_rate,
            highpass,
            lowpass,
            validate=True,
        )
        self.build(kernel=kernel)

    def forward(self, X: TimeSeries3d, crop: bool = True) -> TimeSeries3d:
        """Apply the fitted causal filter.

        Args:
            X:
                Batch of multichannel timeseries with shape ``(B, C, T)``.
            crop:
                If ``True``, remove ``fduration * sample_rate`` samples from
                the left edge. If ``False``, return the full timeseries.

        Returns:
            A ``torch.float32`` tensor with shape ``(B, C, T - L)`` when
            cropped, where ``L = int(fduration * sample_rate)``, or
            ``(B, C, T)`` otherwise.
        """
        _validate_minimum_phase_input(X, num_channels=self.num_channels)
        return _minimum_phase_filter(X, self.kernel, crop)
