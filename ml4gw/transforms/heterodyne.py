from typing import Literal

import torch


class Heterodyne(torch.nn.Module):
    r"""
    Heterodyne transform for time-series data using a single
    chirp mass or a grid of chirp masses.

    This module applies a frequency-domain heterodyne
    transformation to an input timeseries by multiplying its
    Fourier transform with a bank of phase factors corresponding
    to different chirp masses. The result is a set of heterodyned
    signals.

    The heterodyning phase is defined as:

    .. math:: e^{\frac{3i}{128} (\pi \mathcal{M}_c f)^{-5/3}}

    where :math:`\mathcal{M}_c` is the chirp mass and :math:`f` is
    the frequency.

    Args:
        sample_rate (int):
            Sampling rate (Hz) of the input timeseries.
        kernel_length (int):
            Duration (seconds) of the input timeseries segment.
        chirp_mass (float, optional):
            Chirp mass in units of solar masses. If provided,
            a single chirp mass is used for heterodyning.
        num_chirp_masses (int, optional):
            Number of chirp mass templates. Required if
            `chirp_mass` is not provided.
        min_chirp_mass (float, optional):
            Minimum chirp mass (in solar masses) in the grid.
            Required if `chirp_mass` is not provided.
        max_chirp_mass (float, optional):
            Maximum chirp mass (in solar masses) in the grid.
            Required if `chirp_mass` is not provided.
        chirp_mass_distribution (Literal["uniform", "log_uniform"], optional):
            Distribution used to construct the chirp mass grid.
            Required if `chirp_mass` is not provided.
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
        return_type: Literal["time", "freq", "both"],
        num_chirp_masses: int | None = None,
        min_chirp_mass: float | None = None,
        max_chirp_mass: float | None = None,
        chirp_mass_distribution: Literal["uniform", "log_uniform"]
        | None = None,
        chirp_mass: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.freq_grid = self._freq_grid()

        if chirp_mass is not None:
            self.chirp_mass_grid = torch.tensor([chirp_mass])
        else:
            if None in (
                num_chirp_masses,
                min_chirp_mass,
                max_chirp_mass,
                chirp_mass_distribution,
            ):
                raise ValueError(
                    "For chirp mass grid, provide num_chirp_masses,"
                    "min_chirp_mass, max_chirp_mass, and"
                    "chirp_mass_distribution."
                )
            self.num_chirp_masses = num_chirp_masses
            self.min_chirp_mass = min_chirp_mass
            self.max_chirp_mass = max_chirp_mass
            self.chirp_mass_distribution = chirp_mass_distribution
            self.chirp_mass_grid = self._chirp_mass_grid()

        self.pi_m_f = (
            torch.pi * self.chirp_mass_grid[:, None] * self.freq_grid[None, :]
        )
        self.heterodyning_phase = self._heterodyning_phase()
        self.return_type = return_type
        if self.return_type not in {"time", "freq", "both"}:
            raise ValueError(
                "Invalid return_type. Must be one of {'time', 'freq', 'both'}."
            )

    def _freq_grid(self):
        r"""
        Compute the frequency grid for the Fourier transform
        based on the sample rate and kernel length.

        This corresponds to the frequencies for an input timeseries
        of length ``T = sample_rate x kernel_length``.

        Returns:
            torch.Tensor:
                1D tensor of shape `(F,)` where `F = T // 2 + 1`,
                containing frequency bins in Hz.
        """
        freq_grid = torch.fft.rfftfreq(
            self.kernel_length * self.sample_rate, d=1 / self.sample_rate
        )
        return freq_grid

    def _chirp_mass_grid(self):
        r"""
        Compute the chirp mass grid, used for heterodyning, based
        on the specified distribution.

        The grid spans the interval `[min_chirp_mass, max_chirp_mass]`
        using either linear or logarithmic spacing.

        Returns:
            torch.Tensor:
                1D tensor of shape `(M,)` containing chirp masses.
        """
        if self.chirp_mass_distribution == "uniform":
            chirp_mass_grid = torch.linspace(
                self.min_chirp_mass, self.max_chirp_mass, self.num_chirp_masses
            )
        elif self.chirp_mass_distribution == "log_uniform":
            chirp_mass_grid = torch.logspace(
                torch.log10(torch.tensor(self.min_chirp_mass)),
                torch.log10(torch.tensor(self.max_chirp_mass)),
                self.num_chirp_masses,
            )
        else:
            raise ValueError("Invalid chirp mass distribution")
        return chirp_mass_grid

    def _heterodyning_phase(self):
        r"""
        Compute the heterodyning phase for each chirp mass and frequency.

        Returns:
            torch.Tensor:
                Tensor of shape `(M, F)` containing complex phase factors.
        """
        heterodyning_phase = torch.exp((3j / 128) * (self.pi_m_f ** (-5 / 3)))
        return heterodyning_phase

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
