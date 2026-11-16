import torch
from jaxtyping import Float
from torch import Tensor

from ...transforms import SpectralDensity
from ...types import FrequencySeries3d, TimeSeries3d
from ...utils.slicing import unfold_windows


class StreamWhitener(torch.nn.Module):
    """
    Calculate the PSDs and whiten an entire batch of kernels at once.

    Combines PSD estimation and whitening in a single pass, and extracts
    kernels via unfolding. Each stream is split into two unequal-length
    segments, with the first being used to calculate the PSD, and the
    second being whitened.

    Args:
        kernel_length:
            Length, in seconds, of each kernel passed to the model.
        psd_length:
            Length, in seconds, of the data at the start of each
            stream used to estimate the PSD.
        inference_sampling_rate:
            Rate, in Hz, at which kernels are sampled from the
            streams, and so at which the model produces outputs.
            Determines the overlap between kernels.
        batch_size:
            Number of kernels to extract from each stream.
        spectral_density:
            PSD estimator applied to the first segment of each
            stream. Its sample rate must match ``transform``'s.
        transform:
            Whitening module called as ``transform(X, asd)``, such as
            :class:`~ml4gw.transforms.Whiten` or
            :class:`~ml4gw.transforms.MinimumPhaseWhiten`. The sample
            rate and whitening filter length are taken from its
            ``sample_rate`` and ``fduration`` attributes.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from ml4gw.nn.streaming import StreamWhitener
            >>> from ml4gw.transforms import SpectralDensity, Whiten
            >>> whitener = StreamWhitener(
            ...     kernel_length=8, psd_length=64,
            ...     inference_sampling_rate=16, batch_size=128,
            ...     spectral_density=SpectralDensity(
            ...         2048, fftlength=2, average="median", fast=True
            ...     ),
            ...     transform=Whiten(2, 2048, highpass=32),
            ... )
            >>> # (streams, channels, time)
            >>> x = torch.randn(4, 2, whitener.input_size)
            >>> # kernels: (streams * batch_size, channels, kernel_size)
            >>> kernels = whitener(x)
    """

    def __init__(
        self,
        kernel_length: float,
        psd_length: float,
        inference_sampling_rate: float,
        batch_size: int,
        spectral_density: SpectralDensity,
        transform: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.spectral_density = spectral_density
        self.transform = transform
        self.batch_size = batch_size

        if spectral_density.sample_rate != transform.sample_rate:
            raise ValueError(
                "Spectral density sample rate "
                f"{spectral_density.sample_rate} doesn't match whitening "
                f"transform sample rate {transform.sample_rate}"
            )
        self.sample_rate = sample_rate = transform.sample_rate
        fduration = transform.fduration

        # Calculate stride between kernels based on inference sampling rate
        self.stride_size = int(sample_rate / inference_sampling_rate)
        # Convert kernel length to samples
        self.kernel_size = int(kernel_length * sample_rate)

        # do length calculations in units of samples to guard for
        # intification. The data to whiten covers every kernel in the
        # update plus the whitening filter.
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        self.size = strides + self.kernel_size + fsize
        self.psd_size = int(psd_length * sample_rate)
        self.input_size = self.psd_size + self.size

    def whiten(
        self,
        x: Float[Tensor, "stream channel time"],
        psd_source: Tensor | None = None,
    ) -> tuple[TimeSeries3d, FrequencySeries3d]:
        """
        Split data into PSD estimation and whitening segments, then
        whiten the second with the PSD of the first.

        Args:
            x:
                Streams to whiten, with shape
                ``(streams, channels, input_size)``. The first
                ``psd_size`` samples of each stream are used to calculate
                the PSD, and the last ``size`` samples are whitened.
            psd_source:
                ``streams`` indices. The ``i``th stream is whitened by the
                PSD of stream ``psd_source[i]``. If left as ``None``, each
                stream is whitened by its own PSD.

        Returns:
            The whitened streams in the dtype of ``x``, with shape
            ``(streams, channels, size - fduration * sample_rate)``, and
            the ASDs used to whiten them, with shape
            ``(streams, channels, frequency)``.
        """
        # Split data into segments
        background, data = torch.split(x, [self.psd_size, self.size], dim=-1)
        # Use each stream's PSD source's background to calculate its PSD,
        # so that, e.g., data with injected signals can be whitened with
        # the PSD of raw background data
        if psd_source is not None:
            background = background[psd_source]

        # Calculate ASDs from background segment
        asd = self.spectral_density(background.double()) ** 0.5

        # Apply whitening using estimated ASD. The float64 ASD promotes the
        # whitened data to float64, so cast it back to the input's dtype.
        whitened = self.transform(data, asd)
        return whitened.to(x.dtype), asd

    def unfold(
        self, whitened: TimeSeries3d
    ) -> Float[Tensor, "kernel channel time"]:
        """
        Unfold the whitened streams into kernels, and transpose/flatten
        so that the ordering goes stream by stream and different streams'
        outputs aren't interleaved.
        """
        kernels = unfold_windows(
            whitened, self.kernel_size, self.stride_size
        ).transpose(0, 1)
        # Flatten to (streams * batch_size, channels, kernel_size)
        return kernels.flatten(0, 1)

    def forward(
        self,
        x: Float[Tensor, "stream channel time"],
        psd_source: Tensor | None = None,
    ) -> Float[Tensor, "kernel channel time"]:
        """
        Estimate PSD, whiten data, and unfold kernels.

        Subclasses that need the ASD, for example to compute features
        from it, can override this method.

        Args:
            x:
                Streams to whiten, with shape
                ``(streams, channels, input_size)``. The first
                ``psd_size`` samples of each stream are used to calculate
                the PSD, and the last ``size`` samples are whitened.
            psd_source:
                ``streams`` indices. The ``i``th stream is whitened by the
                PSD of stream ``psd_source[i]``. If left as ``None``, each
                stream is whitened by its own PSD.

        Returns:
            Kernels with shape
            ``(streams * batch_size, channels, kernel_size)``, ordered
            stream by stream.
        """
        whitened, _ = self.whiten(x, psd_source)
        return self.unfold(whitened)
