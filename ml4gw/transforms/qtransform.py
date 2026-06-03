import math
import warnings

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..types import TimeSeries1to3d, TimeSeries3d
from .spline_interpolation import SplineInterpolate1D

"""
All based on https://github.com/gwpy/gwpy/blob/v3.0.8/gwpy/signal/qtransform.py
The methods, names, and descriptions come almost entirely from GWpy.
This code allows the Q-transform to be performed on batches of multi-channel
input on GPU.
"""


class _TileGroup(torch.nn.Module):
    """Batched Q-tile computation for one group of QTiles that share the same
    ntiles() value."""

    def __init__(
        self,
        indices: Tensor,
        windows: Tensor,
        scatter: Tensor,
        ntiles_val: int,
        tile_indices: list[int],
    ) -> None:
        super().__init__()
        self.register_buffer("indices", indices)
        self.register_buffer("windows", windows)
        self.register_buffer("scatter", scatter)
        self.ntiles_val = ntiles_val
        self.tile_indices = tile_indices

    def forward(self, X: Tensor, norm: str | None) -> Tensor:
        """
        Args:
            X: rfft output, shape ``(B, C, F)``
            norm: normalization method ("median", "mean", or ``None``).

        Returns:
            Energy tensor of shape ``(B, C, N, ntiles_val)``.
        """
        N = self.indices.shape[0]
        gathered = X[..., self.indices] * self.windows  # (B, C, N, max_ws)

        # Scatter into the ntiles-length buffer. scatter_ is needed because
        # each tile has a different left-padding offset.
        padded = torch.zeros(
            *X.shape[:-1], N, self.ntiles_val, dtype=X.dtype, device=X.device
        )
        padded.scatter_(
            dim=-1,
            index=self.scatter.expand(*X.shape[:-1], -1, -1),
            src=gathered,
        )

        tdenergy = torch.fft.ifft(torch.fft.ifftshift(padded, dim=-1))
        energy = tdenergy.real**2.0 + tdenergy.imag**2.0

        if norm:
            norm_lower = norm.lower()
            if norm_lower == "median":
                energy = energy / torch.quantile(
                    energy, q=0.5, dim=-1, keepdim=True
                )
            elif norm_lower == "mean":
                energy = energy / energy.mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid normalisation {norm}")
            energy = energy.float()
        return energy


def _qtile_geometry(
    q: float,
    frequency: float,
    duration: float,
    sample_rate: float,
    mismatch: float,
) -> tuple[int, int, Tensor, Tensor, int]:
    """Compute the geometry for one Q-tile row."""
    qprime = q / 11**0.5
    deltam = 2 * (mismatch / 3.0) ** 0.5
    windowsize = 2 * int(frequency / qprime * duration) + 1
    tcum_mismatch = duration * 2 * math.pi * frequency / q
    ntiles = int(2 ** math.ceil(math.log2(tcum_mismatch / deltam)))

    half = (windowsize - 1) // 2
    indices = torch.arange(-half, half + 1)

    xfrequencies = (indices / duration) * qprime / frequency
    norm = (
        ntiles
        / (duration * sample_rate)
        * (315 * qprime / (128 * frequency)) ** 0.5
    )
    window = torch.tensor((1 - xfrequencies**2) ** 2 * norm)

    n_rfft = int(duration * sample_rate) // 2
    data_indices = (
        torch.round(indices + 1 + frequency * duration).long().clamp(0, n_rfft)
    )

    left_pad = int((ntiles - windowsize - 1) / 2.0)

    return ntiles, windowsize, window, data_indices, left_pad


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
            ``(num_f_bins, num_t_bins)``. Because the
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
        interpolation_method:
            The method by which to interpolate each ``QTile`` to the specified
            number of time and frequency bins. The acceptable values are
            "bilinear", "bicubic", and "spline". The "bilinear" and "bicubic"
            options will use PyTorch's built-in interpolation modes, while
            "spline" will use the custom Torch-based implementation in
            ``ml4gw``, as PyTorch does not have spline-based intertpolation.
            The "spline" mode is most similar to the results of GWpy's
            Q-transform, which uses ``scipy`` to do spline interpolation.
            However, it is also the slowest and most memory intensive due to
            the matrix equation solving steps. Therefore, the default method
            is "bicubic" as it produces the most similar results while
            optimizing for computing performance.
    """

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        spectrogram_shape: tuple[int, int],
        q: float = 12,
        frange: list[float] = None,
        mismatch: float = 0.2,
        interpolation_method: str = "bicubic",
    ) -> None:
        super().__init__()
        self.q = q
        self.sample_rate = sample_rate
        self.spectrogram_shape = spectrogram_shape
        self.frange = frange or [0, torch.inf]
        self.duration = duration
        self.mismatch = mismatch

        # If q is too large, the minimum of the frange computed
        # below will be larger than the maximum
        max_q = torch.pi * duration * sample_rate / 50 - 11 ** (0.5)
        if q >= max_q:
            raise ValueError(
                "The given q value is too large for the given duration and "
                f"sample rate. The maximum allowable value is {max_q}"
            )

        if interpolation_method not in ["bilinear", "bicubic", "spline"]:
            raise ValueError(
                "Interpolation method must be either 'bilinear', 'bicubic', "
                f"or 'spline'; got {interpolation_method}"
            )
        self.interpolation_method = interpolation_method

        qprime = self.q / 11 ** (1 / 2.0)
        if self.frange[0] <= 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * torch.pi * duration)
        if math.isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = sample_rate / 2 / (1 + 1 / qprime)

        self.freqs = self.get_freqs()
        self.qtiles = None
        self._setup_batched_tile_groups()

        if self.interpolation_method == "spline":
            self._set_up_spline_interp()

    def _setup_batched_tile_groups(self):
        # Group tiles that share the same ntiles value so their IFFTs can be
        # batched in compute_qtiles rather than called one tile at a time.
        geometries = [
            _qtile_geometry(
                self.q,
                float(f),
                self.duration,
                self.sample_rate,
                self.mismatch,
            )
            for f in self.freqs
        ]
        ntiles_list = [g[0] for g in geometries]
        unique_ntiles = sorted(set(ntiles_list))

        groups = []
        for ntiles_val in unique_ntiles:
            tile_indices = [
                i for i, n in enumerate(ntiles_list) if n == ntiles_val
            ]
            N = len(tile_indices)
            max_ws = max(geometries[i][1] for i in tile_indices)

            indices_padded = torch.zeros(N, max_ws, dtype=torch.long)
            windows_padded = torch.zeros(N, max_ws, dtype=torch.float32)
            scatter_idx = torch.zeros(N, max_ws, dtype=torch.long)

            for row, tile_idx in enumerate(tile_indices):
                _, ws, window, data_indices, left_pad = geometries[tile_idx]
                indices_padded[row, :ws] = data_indices
                windows_padded[row, :ws] = window
                scatter_idx[row, :ws] = left_pad + torch.arange(ws)

            groups.append(
                _TileGroup(
                    indices_padded,
                    windows_padded,
                    scatter_idx,
                    ntiles_val,
                    tile_indices,
                )
            )

        self.tile_groups = torch.nn.ModuleList(groups)

    def _set_up_spline_interp(self):
        # tile_groups is already grouped by ntiles_val, so reuse it directly
        self.stack_idx = [group.tile_indices for group in self.tile_groups]

        t_out = torch.arange(
            0, self.duration, self.duration / self.spectrogram_shape[1]
        )
        self.qtile_interpolators = torch.nn.ModuleList(
            [
                SplineInterpolate1D(
                    kx=3,
                    x_in=torch.arange(
                        0, self.duration, self.duration / group.ntiles_val
                    ),
                    x_out=t_out,
                )
                for group in self.tile_groups
            ]
        )

        f_in = self.freqs
        f_out = torch.logspace(
            math.log10(self.frange[0]),
            math.log10(self.frange[-1]),
            self.spectrogram_shape[0],
        )

        self.interpolator = SplineInterpolate1D(
            kx=3,
            x_in=f_in,
            x_out=f_out,
        )

    def get_freqs(self) -> Float[Tensor, " nfreq"]:
        """
        Calculate the frequencies that will be used in this transform.
        For each frequency, a ``QTile`` is created.
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
        # Cast freqs to float64 to avoid off-by-ones from rounding
        freqs = (minf * freqs.double() // fstepmin) * fstepmin
        return torch.unique(freqs)

    def get_max_energy(
        self, fsearch_range: list[float] = None, dimension: str = "both"
    ):
        """
        Gets the maximum energy value among the QTiles. The maximum can
        be computed across all batches and channels, across all channels,
        across all batches, or individually for each channel/batch
        combination. This could be useful for allowing the use of different
        Q values for different channels and batches, but the slicing would
        be slow, so this isn't used yet.

        Optionally, a pair of frequency values can be specified for
        ``fsearch_range`` to restrict the frequencies in which the maximum
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
        for each ``QTile``
        """
        # Computing the FFT with the same normalization and scaling as GWpy
        if X.ndim == 1:
            X = X[None, None]
        elif X.ndim == 2:
            X = X[None]
        X = torch.fft.rfft(X, norm="forward")
        X[..., 1:] *= 2

        results = [None] * len(self.freqs)

        for group in self.tile_groups:
            energy = group(X, norm)  # (B, C, N, ntiles_val)
            for row, tile_idx in enumerate(group.tile_indices):
                results[tile_idx] = energy[..., row, :]

        self.qtiles = results

    def interpolate(self) -> TimeSeries3d:
        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )
        if self.interpolation_method == "spline":
            qtiles = [
                torch.stack([self.qtiles[i] for i in idx], dim=-2)
                for idx in self.stack_idx
            ]
            time_interped = torch.cat(
                [
                    qtile_interpolator(qtile)
                    for qtile, qtile_interpolator in zip(
                        qtiles, self.qtile_interpolators, strict=True
                    )
                ],
                dim=-2,
            )
            # Transpose because the final dimension gets interpolated
            return self.interpolator(time_interped.mT).mT
        num_f_bins, num_t_bins = self.spectrogram_shape
        time_interped = torch.stack(
            [
                F.interpolate(
                    qtile.unsqueeze(0),
                    (qtile.shape[-2], num_t_bins),
                    mode=self.interpolation_method,
                ).squeeze(0)
                for qtile in self.qtiles
            ],
            dim=-2,
        )
        return torch.squeeze(
            F.interpolate(
                time_interped,
                (num_f_bins, num_t_bins),
                mode=self.interpolation_method,
            )
        )

    def forward(
        self,
        X: TimeSeries1to3d,
        norm: str = "median",
    ):
        """
        Compute the Q-tiles and interpolate

        Args:
            X:
                Time series of data. Should have the duration and sample rate
                used to initialize this object. Expected input shape is
                ``(B, C, T)``, where T is the number of samples, C is the
                number of channels, and B is the number of batches. If less
                than three-dimensional, axes will be added during Q-tile
                computation.
            norm:
                The method of normalization used by each QTile

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
            ``(num_f_bins, num_t_bins)``. Because the
            frequency spacing of the Q-tiles is in log-space, the frequency
            interpolation is log-spaced as well.
        qrange:
            The lower and upper values of Q to consider. The
            actual values of Q used for the transforms are
            determined by the ``get_qs`` method
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
        spectrogram_shape: tuple[int, int],
        qrange: list[float] = None,
        frange: list[float] = None,
        interpolation_method="bicubic",
        mismatch: float = 0.2,
    ) -> None:
        super().__init__()
        self.qrange = qrange or [4, 64]
        self.mismatch = mismatch
        self.frange = frange or [0, torch.inf]
        self.spectrogram_shape = spectrogram_shape
        max_q = torch.pi * duration * sample_rate / 50 - 11 ** (0.5)
        self.qs = self.get_qs()
        if self.qs[-1] >= max_q:
            warnings.warn(
                "Some Q values exceed the maximum allowable Q value of "
                f"{max_q}. The list of Q values to be tested in this "
                "scan will be truncated to avoid those values.",
                stacklevel=2,
            )

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
                    interpolation_method=interpolation_method,
                    mismatch=self.mismatch,
                )
                for q in self.qs
                if q < max_q
            ]
        )

    def get_qs(self) -> list[float]:
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
        fsearch_range: list[float] = None,
        norm: str = "median",
    ):
        """
        Compute the set of QTiles for each Q transform and determine which
        has the highest energy value. Interpolate and return the
        corresponding set of tiles.

        Args:
            X:
                Time series of data. Should have the duration and sample rate
                used to initialize this object. Expected input shape is
                ``(B, C, T)``, where T is the number of samples, C is the
                number of channels, and B is the number of batches. If less
                than three-dimensional, axes will be added during Q-tile
                computation.
            fsearch_range:
                The lower and upper frequency values within which to search
                for the maximum energy
            norm:
                The method of interpolation used by each QTile

        Returns:
            An interpolated Q-transform for the batch of data. Output will
            have one more dimension than the input
        """
        for transform in self.q_transforms:
            transform.compute_qtiles(X, norm)
        idx = torch.argmax(
            torch.stack(
                [
                    transform.get_max_energy(fsearch_range=fsearch_range)
                    for transform in self.q_transforms
                ]
            )
        )
        return self.q_transforms[idx].interpolate()
