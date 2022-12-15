from typing import Optional

import torch

from ml4gw.spectral import fast_spectral_density, spectral_density


class SpectralDensity(torch.nn.Module):
    """
    Transform for computing either the power spectral density
    of a batch of multichannel timeseries, or the cross spectral
    density of two batches of multichannel timeseries.

    On `SpectralDensity.forward` call, if only one tensor is provided,
    this transform will compute its power spectral density. If a second
    tensor is provided, the cross spectral density between the two
    timeseries will be computed. For information about the allowed
    relationships between these two tensors, see the documentation to
    `ml4gw.spectral.fast_spectral_density`.

    Note that the cross spectral density computation is currently
    only available for the `fast_spectral_density` option. If
    `fast=False` and a second tensor is passed to `SpectralDensity.forward`,
    a `NotImplementedError` will be raised.

    Args:
        sample_rate:
            Rate at which tensors passed to `forward` will be sampled
        fftlength:
            Length of the window, in seconds, to use for FFT estimates
        overlap:
            Overlap between windows used for FFT calculation. If left
            as `None`, this will be set to `fftlength / 2`.
        average:
            Aggregation method to use for combining windowed FFTs.
            Allowed values are `"mean"` and `"median"`.
        fast:
            Whether to use a faster spectral density computation that
            support cross spectral density, or a slower one which does
            not. The cost of the fast implementation is that it is not
            exact for the two lowest frequency bins.
    """

    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        fast: bool = False,
    ) -> None:
        if overlap is None:
            overlap = fftlength / 2
        elif overlap >= fftlength:
            raise ValueError(
                "Can't have overlap {} longer than fftlength {}".format(
                    overlap, fftlength
                )
            )

        super().__init__()

        self.nperseg = int(fftlength * sample_rate)
        self.nstride = self.nperseg - int(overlap * sample_rate)

        # TODOs: Do we allow for arbitrary windows?
        # Making this buffer persistent in case we want
        # to implement this down the line, so that custom
        # windows can be loaded in.
        self.register_buffer("window", torch.hann_window(self.nperseg))

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

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
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
