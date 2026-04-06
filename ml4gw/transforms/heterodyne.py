from typing import Literal

import torch

from ml4gw.constants import MTSUN_SI


class Heterodyne(torch.nn.Module):
    r"""
    Heterodyne transform for time-series data using a single
    chirp mass or a grid of chirp masses.

    This module applies a frequency-domain heterodyne
    transformation to an input timeseries by multiplying its
    Fourier transform with a bank of phase factors corresponding
    to different chirp masses. The result is a set of heterodyned
    signals.

    The heterodyning phase is defined as the leading-order (0PN) term
    in the post-Newtonian expansion of the inspiral phase:

    .. math:: e^{\frac{3i}{128} (\pi \mathcal{M}_c f)^{-5/3}}

    where :math:`\mathcal{M}_c` is the chirp mass and :math:`f` is
    the frequency.

    .. note::
        This uses only the 0PN phase term. Higher-order PN corrections
        are not included in this heterodyne transform.

    Args:
        sample_rate (int):
            Sampling rate (Hz) of the input timeseries.
        kernel_length (int):
            Duration (seconds) of the input timeseries segment.
        chirp_mass (torch.Tensor):
            1D tensor of chirp mass(es) in units of solar masses. The
            shape should be `(M,)` where `M` is the number of masses.
        return_type (Literal["time", "freq", "both"]):
            Specifies whether to return -
                - "time": heterodyned time-domain signals
                - "freq": heterodyned frequency-domain signals
                - "both": tuple of (time, frequency) representations

    Shape:
        - Input: `(B, C, T)` where
            - B = batch size
            - C = number of channels (e.g., detectors for GW strain)
            - T = number of time samples (= sample_rate * kernel_length)

        - Frequency-domain intermediate:
            `(B, C, F)` where F = T // 2 + 1

        - After heterodyning:
            `(B, C, M, F)` where M = number of chirp masses

        - Output:
            - If `return_type=time` → `(B, C, M, T)`
            - If `return_type=freq` → `(B, C, M, F)`
            - If `return_type=both` → tuple: `(B, C, M, T)`, `(B, C, M, F)`

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
            Heterodyned signals in the requested domain(s).
    """

    def __init__(
        self,
        sample_rate: int,
        kernel_length: int,
        chirp_mass: torch.Tensor,
        return_type: Literal["time", "freq", "both"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.chirp_mass = chirp_mass

        self.freq_grid = torch.fft.rfftfreq(
            self.kernel_length * self.sample_rate, d=1 / self.sample_rate
        )

        self.pi_m_f = (
            torch.pi
            * (self.chirp_mass[:, None] * MTSUN_SI)
            * self.freq_grid[None, :]
        )

        self.register_buffer("heterodyning_phase", self._heterodyning_phase())

        self.return_type = return_type
        if self.return_type not in {"time", "freq", "both"}:
            raise ValueError(
                "Invalid return_type. Must be one of {'time', 'freq', 'both'}."
            )

    def _heterodyning_phase(self):
        r"""
        Compute the heterodyning phase for each chirp mass and frequency.

        Returns:
            torch.Tensor:
                Tensor of shape `(M, F)` containing complex phase factors.
        """
        phase = torch.exp((3j / 128) * (self.pi_m_f ** (-5 / 3)))
        return phase

    def forward(self, X: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        r"""
        Apply the heterodyne transformation to the input timeseries.

        Args:
            X (torch.Tensor):
                Input tensor of shape `(B, C, T)`.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If ``return_type=time`` → `(B, C, M, T)`
                - If ``return_type=freq`` → `(B, C, M, F)`
                - If ``return_type=both`` → `(time, freq)`
        """

        X_fft = torch.fft.rfft(X, dim=-1)
        X_fft /= self.sample_rate
        X_heterodyned = X_fft[:, :, None] * self.heterodyning_phase[None, :, :]
        X_heterodyned[..., 0] = 0
        X_ifft = torch.fft.irfft(X_heterodyned, dim=-1)
        X_ifft *= self.sample_rate

        if self.return_type == "time":
            return X_ifft
        elif self.return_type == "freq":
            return X_heterodyned
        else:
            return X_ifft, X_heterodyned
