import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from ml4gw.transforms.spline_interpolation import SplineInterpolate
from ml4gw.types import FrequencySeries1to3d, TimeSeries1to3d, TimeSeries3d

"""
All based on https://github.com/gwpy/gwpy/blob/v3.0.8/gwpy/signal/qtransform.py
The methods, names, and descriptions come almost entirely from GWpy.
This code allows the Q-transform to be performed on batches of multi-channel
input on GPU.
"""


class QTile(torch.nn.Module):
    """
    Compute the row of Q-tiles for a single Q value and a single
    frequency for a batch of multi-channel frequency series data.
    Should really be called `QRow`, but I want to match GWpy.
    Input data should have three dimensions or fewer.
    If fewer, dimensions will be added until the input is
    three-dimensional.

    Args:
        q:
            The Q value to use in computing the Q tile
        frequency:
            The frequency for which to compute the Q tile in Hz
        duration:
            The length of time in seconds that the input frequency
            series represents
        sample_rate:
            The sample rate of the original time series in Hz
        mismatch:
            The maximum fractional mismatch between neighboring tiles

    """

    def __init__(
        self,
        q: float,
        frequency: float,
        duration: float,
        sample_rate: float,
        mismatch: float,
    ) -> None:
        super().__init__()
        self.mismatch = mismatch
        self.q = q
        self.deltam = torch.tensor(2 * (self.mismatch / 3.0) ** (1 / 2.0))
        self.qprime = self.q / 11 ** (1 / 2.0)
        self.frequency = frequency
        self.duration = duration
        self.sample_rate = sample_rate

        self.windowsize = (
            2 * int(self.frequency / self.qprime * self.duration) + 1
        )
        pad = self.ntiles() - self.windowsize
        padding = torch.Tensor((int((pad - 1) / 2.0), int((pad + 1) / 2.0)))
        self.register_buffer("padding", padding)
        self.register_buffer("indices", self.get_data_indices())
        self.register_buffer("window", self.get_window())

    def ntiles(self) -> int:
        """
        Number of tiles in this frequency row
        """
        tcum_mismatch = self.duration * 2 * torch.pi * self.frequency / self.q
        return int(2 ** torch.ceil(torch.log2(tcum_mismatch / self.deltam)))

    def _get_indices(self) -> Int[Tensor, " windowsize"]:
        half = int((self.windowsize - 1) / 2)
        return torch.arange(-half, half + 1)

    def get_window(self) -> Float[Tensor, " windowsize"]:
        """
        Generate the bi-square window for this row
        """
        wfrequencies = self._get_indices() / self.duration
        xfrequencies = wfrequencies * self.qprime / self.frequency
        norm = (
            self.ntiles()
            / (self.duration * self.sample_rate)
            * (315 * self.qprime / (128 * self.frequency)) ** (1 / 2.0)
        )
        return torch.Tensor((1 - xfrequencies**2) ** 2 * norm)

    def get_data_indices(self) -> Int[Tensor, " windowsize"]:
        """
        Get the index array of relevant frequencies for this row
        """
        return torch.round(
            self._get_indices() + 1 + self.frequency * self.duration,
        ).type(torch.long)

    def forward(
        self,
        fseries: FrequencySeries1to3d,
        norm: str = "median",
    ) -> TimeSeries1to3d:
        """
        Compute the transform for this row

        Args:
            fseries:
                Frequency series of data. Should correspond to data with
                the duration and sample rate used to initialize this object.
                Expected input shape is `(B, C, F)`, where F is the number
                of samples, C is the number of channels, and B is the number
                of batches. If less than three-dimensional, axes will be
                added.
            norm:
                The method of normalization. Options are "median", "mean", or
                `None`.

        Returns:
            The row of Q-tiles for the given Q and frequency. Output is
            three-dimensional: `(B, C, T)`
        """
        if len(fseries.shape) > 3:
            raise ValueError("Input data has more than 3 dimensions")

        while len(fseries.shape) < 3:
            fseries = fseries[None]

        windowed = fseries[..., self.indices] * self.window
        left, right = self.padding
        padded = F.pad(windowed, (int(left), int(right)), mode="constant")
        wenergy = torch.fft.ifftshift(padded, dim=-1)

        tdenergy = torch.fft.ifft(wenergy)
        energy = tdenergy.real**2.0 + tdenergy.imag**2.0
        if norm:
            norm = norm.lower() if isinstance(norm, str) else norm
            if norm == "median":
                medians = torch.quantile(energy, q=0.5, dim=-1, keepdim=True)
                energy /= medians
            elif norm == "mean":
                means = torch.mean(energy, dim=-1, keepdim=True)
                energy /= means
            else:
                raise ValueError("Invalid normalisation %r" % norm)
            energy = energy.type(torch.float32)
        return energy


class SingleQTransform(torch.nn.Module):
    """
    Compute the Q-transform for a single Q value for a batch of
    multi-channel time series data. Input data should have
    three dimensions or fewer.

    Args:
        duration:
            Length of the time series data in seconds
        sample_rate:
            Sample rate of the data in Hz
        spectrogram_shape:
            The shape of the interpolated spectrogram, specified as
            `(num_f_bins, num_t_bins)`. Because the
            frequency spacing of the Q-tiles is in log-space, the frequency
            interpolation is log-spaced as well.
        q:
            The Q value to use for the Q transform
        frange:
            The lower and upper frequency limit to consider for
            the transform. If unspecified, default values will
            be chosen based on q, sample_rate, and duration
        mismatch:
            The maximum fractional mismatch between neighboring tiles
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        spectrogram_shape: Tuple[int, int],
        q: float = 12,
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ) -> None:
        super().__init__()
        self.q = q
        self.spectrogram_shape = spectrogram_shape
        self.frange = frange
        self.duration = duration
        self.mismatch = mismatch

        qprime = self.q / 11 ** (1 / 2.0)
        if self.frange[0] <= 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * torch.pi * duration)
        if math.isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = sample_rate / 2 / (1 + 1 / qprime)
        self.freqs = self.get_freqs()
        self.qtile_transforms = torch.nn.ModuleList(
            [
                QTile(
                    q=self.q,
                    frequency=freq,
                    duration=self.duration,
                    sample_rate=sample_rate,
                    mismatch=self.mismatch,
                )
                for freq in self.freqs
            ]
        )
        self.qtiles = None

        self.qtile_interpolators = torch.nn.ModuleList(
            [
                SplineInterpolate(
                    kx=3,
                    x_in=torch.linspace(-1, 1, qtile.ntiles()),
                    y_in=torch.linspace(-1, 1, 1),
                    x_out=torch.linspace(-1, 1, spectrogram_shape[1]),
                    y_out=torch.linspace(-1, 1, 1),
                )
                for qtile in self.qtile_transforms
            ]
        )

        t_in = t_out = torch.linspace(-1, 1, spectrogram_shape[1])
        f_in = self.freqs
        f_out = torch.logspace(
            math.log10(f_in[0]), math.log10(f_in[-1]), spectrogram_shape[0]
        )

        self.interpolator = SplineInterpolate(
            kx=3,
            ky=3,
            x_in=t_in,
            y_in=f_in,
            x_out=t_out,
            y_out=f_out,
        )

    def get_freqs(self) -> Float[Tensor, " nfreq"]:
        """
        Calculate the frequencies that will be used in this transform.
        For each frequency, a `QTile` is created.
        """
        minf, maxf = self.frange
        fcum_mismatch = (
            math.log(maxf / minf) * (2 + self.q**2) ** (1 / 2.0) / 2.0
        )
        deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        nfreq = int(max(1, math.ceil(fcum_mismatch / deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1 / self.duration

        freq_base = math.exp(2 / ((2 + self.q**2) ** (1 / 2.0)) * fstep)
        freqs = torch.Tensor([freq_base ** (i + 0.5) for i in range(nfreq)])
        freqs = (minf * freqs // fstepmin) * fstepmin
        return torch.unique(freqs)

    def get_max_energy(
        self, fsearch_range: List[float] = None, dimension: str = "both"
    ):
        """
        Gets the maximum energy value among the QTiles. The maximum can
        be computed across all batches and channels, across all channels,
        across all batches, or individually for each channel/batch
        combination. This could be useful for allowing the use of different
        Q values for different channels and batches, but the slicing would
        be slow, so this isn't used yet.

        Optionally, a pair of frequency values can be specified for
        `fsearch_range` to restrict the frequencies in which the maximum
        energy value is sought.
        """
        allowed_dimensions = ["both", "neither", "channel", "batch"]
        if dimension not in allowed_dimensions:
            raise ValueError(f"Dimension must be one of {allowed_dimensions}")

        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )

        if fsearch_range is not None:
            start = min(torch.argwhere(self.freqs > fsearch_range[0]))
            stop = min(torch.argwhere(self.freqs > fsearch_range[1]))
            qtiles = self.qtiles[start:stop]
        else:
            qtiles = self.qtiles

        if dimension == "both":
            return max([torch.max(qtile) for qtile in qtiles])

        max_across_t = [torch.max(qtile, dim=-1).values for qtile in qtiles]
        max_across_t = torch.stack(max_across_t, dim=-1)
        max_across_ft = torch.max(max_across_t, dim=-1).values

        if dimension == "neither":
            return max_across_ft
        if dimension == "channel":
            return torch.max(max_across_ft, dim=-2).values
        if dimension == "batch":
            return torch.max(max_across_ft, dim=-1).values

    def compute_qtiles(
        self,
        X: TimeSeries1to3d,
        norm: str = "median",
    ) -> None:
        """
        Take the FFT of the input timeseries and calculate the transform
        for each `QTile`
        """
        # Computing the FFT with the same normalization and scaling as GWpy
        X = torch.fft.rfft(X, norm="forward")
        X[..., 1:] *= 2
        self.qtiles = [qtile(X, norm) for qtile in self.qtile_transforms]

    def interpolate(self) -> TimeSeries3d:
        """
        Interpolate each `QTile` to the specified number of time and
        frequency bins. Note that PyTorch does not have the same
        interpolation methods that GWpy uses, and so the interpolated
        spectrograms will be different even though the uninterpolated
        values match. The `bicubic` interpolation method is used as
        it seems to match GWpy most closely.
        """
        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )
        time_interped = torch.stack(
            [
                interpolator(qtile)
                for qtile, interpolator in zip(
                    self.qtiles, self.qtile_interpolators
                )
            ],
            dim=-2,
        )
        return self.interpolator(time_interped.squeeze())

    def forward(
        self,
        X: TimeSeries1to3d,
        norm: str = "median",
        spectrogram_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Compute the Q-tiles and interpolate

        Args:
            X:
                Time series of data. Should have the duration and sample rate
                used to initialize this object. Expected input shape is
                `(B, C, T)`, where T is the number of samples, C is the number
                of channels, and B is the number of batches. If less than
                three-dimensional, axes will be added during Q-tile
                computation.
            norm:
                The method of interpolation used by each QTile
            spectrogram_shape:
                The shape of the interpolated spectrogram, specified as
                `(num_f_bins, num_t_bins)`. Because the
                frequency spacing of the Q-tiles is in log-space, the frequency
                interpolation is log-spaced as well. If not given, the shape
                used to initialize the transform will be used.

        Returns:
            The interpolated Q-transform for the batch of data. Output will
            have one more dimension than the input
        """

        self.compute_qtiles(X, norm)
        return self.interpolate()


class QScan(torch.nn.Module):
    """
    Calculate the Q-transform of a batch of multi-channel
    time series data for a range of Q values and return
    the interpolated Q-transform with the highest energy.

    Args:
        duration:
            Length of the time series data in seconds
        sample_rate:
            Sample rate of the data in Hz
        spectrogram_shape:
            The shape of the interpolated spectrogram, specified as
            `(num_f_bins, num_t_bins)`. Because the
            frequency spacing of the Q-tiles is in log-space, the frequency
            interpolation is log-spaced as well.
        qrange:
            The lower and upper values of Q to consider. The
            actual values of Q used for the transforms are
            determined by the `get_qs` method
        frange:
            The lower and upper frequency limit to consider for
            the transform. If unspecified, default values will
            be chosen based on q, sample_rate, and duration
        mismatch:
            The maximum fractional mismatch between neighboring tiles
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        spectrogram_shape: Tuple[int, int],
        qrange: List[float] = [4, 64],
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ) -> None:
        super().__init__()
        self.qrange = qrange
        self.mismatch = mismatch
        self.qs = self.get_qs()
        self.frange = frange
        self.spectrogram_shape = spectrogram_shape

        # Deliberately doing something different from GWpy here.
        # Their final frange is the intersection of the frange
        # from each q. This implementation uses the frange of
        # the chosen q.
        self.q_transforms = torch.nn.ModuleList(
            [
                SingleQTransform(
                    duration=duration,
                    sample_rate=sample_rate,
                    spectrogram_shape=spectrogram_shape,
                    q=q,
                    frange=self.frange.copy(),
                    mismatch=self.mismatch,
                )
                for q in self.qs
            ]
        )

    def get_qs(self) -> List[float]:
        """
        Determine the values of Q to try for the set of Q-transforms
        """
        deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        cumum = math.log(self.qrange[1] / self.qrange[0]) / 2 ** (1 / 2.0)
        nplanes = int(max(math.ceil(cumum / deltam), 1))
        dq = cumum / nplanes
        qs = [
            self.qrange[0] * math.exp(2 ** (1 / 2.0) * dq * (i + 0.5))
            for i in range(nplanes)
        ]
        return qs

    def forward(
        self,
        X: TimeSeries1to3d,
        fsearch_range: List[float] = None,
        norm: str = "median",
        spectrogram_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Compute the set of QTiles for each Q transform and determine which
        has the highest energy value. Interpolate and return the
        corresponding set of tiles.

        Args:
            X:
                Time series of data. Should have the duration and sample rate
                used to initialize this object. Expected input shape is
                `(B, C, T)`, where T is the number of samples, C is the number
                of channels, and B is the number of batches. If less than
                three-dimensional, axes will be added during Q-tile
                computation.
            fsearch_range:
                The lower and upper frequency values within which to search
                for the maximum energy
            norm:
                The method of interpolation used by each QTile
            spectrogram_shape:
                The shape of the interpolated spectrogram, specified as
                `(num_f_bins, num_t_bins)`. Because the
                frequency spacing of the Q-tiles is in log-space, the frequency
                interpolation is log-spaced as well. If not given, the shape
                used to initialize the transform will be used.

        Returns:
            An interpolated Q-transform for the batch of data. Output will
            have one more dimension than the input
        """
        for transform in self.q_transforms:
            transform.compute_qtiles(X, norm)
        idx = torch.argmax(
            torch.Tensor(
                [
                    transform.get_max_energy(fsearch_range=fsearch_range)
                    for transform in self.q_transforms
                ]
            )
        )
        if spectrogram_shape is None:
            spectrogram_shape = self.spectrogram_shape
        num_f_bins, num_t_bins = spectrogram_shape
        return self.q_transforms[idx].interpolate(num_f_bins, num_t_bins)
