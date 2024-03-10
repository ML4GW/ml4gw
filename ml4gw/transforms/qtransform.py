import math
from typing import List

import torch
import torch.nn.functional as F

"""
All based on https://github.com/gwpy/gwpy/blob/v3.0.8/gwpy/signal/qtransform.py
"""


class QTile(torch.nn.Module):
    def __init__(
        self,
        q: float,
        frequency: float,
        duration: float,
        sample_rate: float,
        mismatch: float,
    ):
        super().__init__()
        self.mismatch = mismatch
        self.q = float(q)
        self.deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
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

    def ntiles(self):
        tcum_mismatch = self.duration * 2 * torch.pi * self.frequency / self.q
        return int(2 ** torch.ceil(torch.log2(tcum_mismatch / self.deltam)))

    def _get_indices(self):
        half = int((self.windowsize - 1) / 2)
        return torch.arange(-half, half + 1)

    def get_window(self):
        wfrequencies = self._get_indices() / self.duration
        xfrequencies = wfrequencies * self.qprime / self.frequency
        norm = (
            self.ntiles()
            / (self.duration * self.sample_rate)
            * (315 * self.qprime / (128 * self.frequency)) ** (1 / 2.0)
        )
        return torch.Tensor((1 - xfrequencies**2) ** 2 * norm)

    def get_data_indices(self):
        return torch.round(
            self._get_indices() + 1 + self.frequency * self.duration,
        ).type(torch.long)

    def forward(self, fseries: torch.Tensor, norm: str = "median"):
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
                medians = torch.median(energy, dim=-1).values
                medians = medians.repeat(energy.shape[-1], 1, 1)
                medians = medians.transpose(0, -1).transpose(0, 1)
                energy /= medians
            elif norm == "mean":
                means = torch.mean(energy, dim=-1)
                means = means.repeat(energy.shape[-1], 1, 1)
                means = means.transpose(0, -1).transpose(0, 1)
                energy /= means
            else:
                raise ValueError("Invalid normalisation %r" % norm)
            return energy.type(torch.float32)
        return energy


class SingleQTransform(torch.nn.Module):
    def __init__(
        self,
        duration: float,
        sample_rate: float,
        q: float = 12,
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ):
        super().__init__()
        self.q = q
        self.frange = frange
        self.duration = duration
        self.mismatch = mismatch

        # This needs to account for whatever highpassing was done for whitening
        # Possibly include whitening in this module?
        qprime = self.q / 11 ** (1 / 2.0)
        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * torch.pi * duration)
        if math.isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = sample_rate / 2 / (1 + 1 / qprime)
        self.freqs = self.get_freqs()
        self.qtiles_transforms = torch.nn.ModuleList(
            [
                QTile(self.q, freq, self.duration, sample_rate, self.mismatch)
                for freq in self.freqs
            ]
        )
        self.qtiles = None

    def get_freqs(self):
        minf, maxf = self.frange
        fcum_mismatch = (
            math.log(maxf / minf) * (2 + self.q**2) ** (1 / 2.0) / 2.0
        )
        deltam = 2 * (self.mismatch / 3.0) ** (1 / 2.0)
        nfreq = int(max(1, math.ceil(fcum_mismatch / deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1 / self.duration

        freq_base = math.exp(2 / ((2 + self.q**2) ** (1 / 2.0)) * fstep)
        freqs = freq_base ** torch.Tensor([i + 0.5 for i in range(nfreq)])
        freqs = (minf * freqs // fstepmin) * fstepmin
        return torch.unique(freqs)

    def get_max_energy(
        self, fsearch_range: List[float] = None, dimension: str = "both"
    ):
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
            return torch.max(max_across_ft, dim=0).values
        if dimension == "batch":
            return torch.max(max_across_ft, dim=-1).values

    def compute_qtiles(self, X: torch.Tensor, norm: str = "median"):
        X = torch.fft.rfft(X, norm="forward")
        X[..., 1:] *= 2
        self.qtiles = [qtile(X, norm) for qtile in self.qtiles_transforms]

    def interpolate(self, num_f_bins: int, num_t_bins: int):
        if self.qtiles is None:
            raise RuntimeError(
                "Q-tiles must first be computed with .compute_qtiles()"
            )
        resampled = [
            F.interpolate(qtile, num_t_bins, mode="linear")
            for qtile in self.qtiles
        ]
        resampled = torch.stack(resampled, dim=-2)
        resampled = F.interpolate(
            resampled, (num_f_bins, num_t_bins), mode="bilinear"
        )
        return torch.squeeze(resampled)

    def forward(
        self,
        X: torch.Tensor,
        num_f_bins: int,
        num_t_bins: int,
        norm: str = "median",
    ):
        self.compute_qtiles(X, norm)
        return self.interpolate(num_f_bins, num_t_bins)


class QScan(torch.nn.Module):
    def __init__(
        self,
        duration: float,
        sample_rate: float,
        qrange: List[float] = [4, 64],
        frange: List[float] = [0, torch.inf],
        mismatch: float = 0.2,
    ):
        super().__init__()
        self.qrange = qrange
        self.mismatch = mismatch
        self.qs = self.get_qs()
        self.frange = frange

        # Deliberately doing something different from GWpy here
        # Their final frange is the intersection of the frange
        # from each q. This implementation uses the frange of
        # the chosen q.
        self.q_transforms = torch.nn.ModuleList(
            [
                SingleQTransform(
                    duration=duration,
                    sample_rate=sample_rate,
                    q=q,
                    frange=self.frange.copy(),
                    mismatch=self.mismatch,
                )
                for q in self.qs
            ]
        )

    def get_qs(self):
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
        X: torch.Tensor,
        num_f_bins: int,
        num_t_bins: int,
        fsearch_range: List[float],
        norm: str = "median",
    ):
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
        return (
            self.q_transforms[idx].interpolate(num_f_bins, num_t_bins),
            self.qs[idx],
        )
