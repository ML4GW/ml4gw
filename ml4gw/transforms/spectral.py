import torch
from jaxtyping import Float
from torch import Tensor

from ..spectral import fast_spectral_density, spectral_density
from ..types import FrequencySeries1to3d, TimeSeries1to3d


class SpectralDensity(torch.nn.Module):
    """
    Transform for computing either the power spectral density
    of a batch of multichannel timeseries, or the cross spectral
    density of two batches of multichannel timeseries.

    On ``SpectralDensity.forward`` call, if only one tensor is provided,
    this transform will compute its power spectral density. If a second
    tensor is provided, the cross spectral density between the two
    timeseries will be computed. For information about the allowed
    relationships between these two tensors, see the documentation to
    :meth:`~ml4gw.spectral.fast_spectral_density`.

    Note that the cross spectral density computation is currently
    only available for :meth:`~ml4gw.spectral.fast_spectral_density`. If
    ``fast=False`` and a second tensor is passed to ``SpectralDensity.forward``,  # noqa E501
    a ``NotImplementedError`` will be raised.

    Args:
        sample_rate:
            Rate at which tensors passed to ``forward`` will be sampled
        fftlength:
            Length of the window, in seconds, to use for FFT estimates
        overlap:
            Overlap between windows used for FFT calculation. If left
            as ``None``, this will be set to ``fftlength / 2``.
        average:
            Aggregation method to use for combining windowed FFTs.
            Allowed values are ``"mean"`` and ``"median"``.
        window:
            Window array to multiply by each FFT window before
            FFT computation. Should have length ``nperseg``.
            Defaults to a hanning window.
        fast:
            Whether to use a faster spectral density computation that
            support cross spectral density, or a slower one which does
            not. The cost of the fast implementation is that it is not
            exact for the two lowest frequency bins.
    """  # noqa E501

    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: float | None = None,
        average: str = "mean",
        window: Float[Tensor, " {int(fftlength*sample_rate)}"] | None = None,
        fast: bool = False,
    ) -> None:
        if overlap is None:
            overlap = fftlength / 2
        elif overlap >= fftlength:
            raise ValueError(
                f"Can't have overlap {overlap} longer than fftlength "
                f"{fftlength}"
            )

        super().__init__()

        self.nperseg = int(fftlength * sample_rate)
        self.nstride = self.nperseg - int(overlap * sample_rate)

        # if no window is provided, default to a hanning window;
        # validate that window is correct size
        if window is None:
            window = torch.hann_window(self.nperseg)

        if window.size(0) != self.nperseg:
            raise ValueError(
                f"Window must have length {self.nperseg} got {window.size(0)}"
            )
        self.register_buffer("window", window)

        # scale corresponds to "density" normalization, worth
        # considering adding this as a kwarg and changing this calc
        scale = 1.0 / (sample_rate * (self.window**2).sum())
        self.register_buffer("scale", scale)

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        self.fast = fast

    def forward(
        self, x: TimeSeries1to3d, y: TimeSeries1to3d | None = None
    ) -> FrequencySeries1to3d:
        if self.fast:
            return fast_spectral_density(
                x,
                y=y,
                nperseg=self.nperseg,
                nstride=self.nstride,
                window=self.window,
                scale=self.scale,
                average=self.average,
            )

        if y is not None:
            raise NotImplementedError
        return spectral_density(
            x,
            nperseg=self.nperseg,
            nstride=self.nstride,
            window=self.window,
            scale=self.scale,
            average=self.average,
        )
