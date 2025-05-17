import torch
from torch import Tensor

from ml4gw.types import BatchTensor
from typing import Tuple, Dict

def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - (e * e))
    b = a * torch.sqrt(1.0 - (e * e))
    return a, b


class SineGaussian(torch.nn.Module):
    """
    Callable class for generating sine-Gaussian waveforms.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        # determine times based on requested duration and sample rate
        # and shift so that the waveform is centered at t=0

        num = int(duration * sample_rate)
        times = torch.arange(num, dtype=torch.float64) / sample_rate
        times -= duration / 2.0

        self.register_buffer("times", times)

    def forward(
        self,
        quality: BatchTensor,
        frequency: BatchTensor,
        hrss: BatchTensor,
        phase: BatchTensor,
        eccentricity: BatchTensor,
    ):
        """
        Generate lalinference implementation of a sine-Gaussian waveform.
        See
        git.ligo.org/lscsoft/lalsuite/-/blob/master/lalinference/lib/LALInferenceBurstRoutines.c#L381
        for details on parameter definitions.

        Args:
            frequency:
                Central frequency of the sine-Gaussian waveform
            quality:
                Quality factor of the sine-Gaussian waveform
            hrss:
                Hrss of the sine-Gaussian waveform
            phase:
                Phase of the sine-Gaussian waveform
            eccentricity:
                Eccentricity of the sine-Gaussian waveform.
                Controls the relative amplitudes of the
                hplus and hcross polarizations.
        Returns:
            Tensors of cross and plus polarizations
        """
        dtype = frequency.dtype
        # add dimension for calculating waveforms in batch
        frequency = frequency.view(-1, 1)
        quality = quality.view(-1, 1)
        hrss = hrss.view(-1, 1)
        phase = phase.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)

        # TODO: enforce all inputs are on the same device?
        pi = torch.tensor([torch.pi], device=frequency.device)

        # calculate relative hplus / hcross amplitudes based on eccentricity
        # as well as normalization factors
        a, b = semi_major_minor_from_e(eccentricity)
        norm_prefactor = quality / (4.0 * frequency * torch.sqrt(pi))
        cosine_norm = norm_prefactor * (1.0 + torch.exp(-quality * quality))
        sine_norm = norm_prefactor * (1.0 - torch.exp(-quality * quality))

        cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)

        h0_plus = (
            hrss
            * a
            / torch.sqrt(
                cosine_norm * (cos_phase**2) + sine_norm * (sin_phase**2)
            )
        )
        h0_cross = (
            hrss
            * b
            / torch.sqrt(
                cosine_norm * (sin_phase**2) + sine_norm * (cos_phase**2)
            )
        )

        # cast the phase to a complex number
        phi = 2 * pi * frequency * self.times
        complex_phase = torch.complex(torch.zeros_like(phi), (phi - phase))

        # calculate the waveform and apply a tukey window to taper the waveform
        fac = torch.exp(phi**2 / (-2.0 * quality**2) + complex_phase)

        cross = fac.imag * h0_cross
        plus = fac.real * h0_plus

        cross = cross.to(dtype)
        plus = plus.to(dtype)

        return cross, plus

import torch
from typing import Tuple

class MultiSineGaussian(torch.nn.Module):
    """
    Generate a sum of ≤10 sine-Gaussians with small time offsets so they
    don't pile up at exactly t=0.

    Parameters
    ----------
    sample_rate : float
    duration    : float
    n_max       : int   (fixed at 10 here)
    max_shift   : float (seconds)  maximum absolute time-offset per burst
    """

    def __init__(
        self,
        sample_rate : float,
        duration    : float,
        n_max       : int   = 10,
        max_shift   : float = 100e-3,     # 100 ms
    ):
        super().__init__()
        self.sg         = SineGaussian(sample_rate, duration)
        self.n_max      = n_max
        self.max_shift  = max_shift
        self.sample_rate= sample_rate

        # expose sample times if needed
        self.register_buffer("times", self.sg.times, persistent=False)

    # ------------------------------------------------------------------
    def _roll_batch(self, x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """
        Vectorised row-wise circular shift.
        x      : (M, N)
        shifts : (M,)  integer samples; positive → shift right
        """
        M, N = x.shape
        # Indices matrix: (M, N)
        idx = (torch.arange(N, device=x.device).view(1, N) - shifts.view(-1, 1)) % N
        return x.gather(1, idx)

    # ------------------------------------------------------------------
    def forward(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.times.device
        batch  = kwargs["n_components"].shape[0]
        dtype  = kwargs["quality_1"].dtype

        # ---------- 1) stack parameters  -------------------------------
        def stack(name):
            return torch.stack(
                [kwargs[f"{name}_{i}"].to(device) for i in range(1, self.n_max + 1)],
                dim=1,
            )

        quality      = stack("quality")         # (B, n_max)
        frequency    = stack("frequency")
        hrss         = stack("hrss")
        phase        = stack("phase")
        eccentricity = stack("eccentricity")

        n_comp = kwargs["n_components"].to(device).unsqueeze(-1)   # (B,1)

        # ---------- 2) mask for active components ----------------------
        mask = (
            torch.arange(self.n_max, device=device)
                  .unsqueeze(0) < n_comp
        ).unsqueeze(-1)                            # (B, n_max, 1)

        # ---------- 3) flatten active params --------------------------
        active = mask.squeeze(-1)                  # (B, n_max) boolean
        quality_f      = quality     [active]
        frequency_f    = frequency   [active]
        hrss_f         = hrss        [active]
        phase_f        = phase       [active]
        eccentricity_f = eccentricity[active]

        if quality_f.numel() == 0:                 # safety net
            zeros = torch.zeros(batch, self.times.numel(), dtype=dtype, device=device)
            return zeros, zeros

        # ---------- 4) generate raw bursts ----------------------------
        cross_f, plus_f = self.sg(
            quality      = quality_f,
            frequency    = frequency_f,
            hrss         = hrss_f,
            phase        = phase_f,
            eccentricity = eccentricity_f,
        )                                           # (M, N)

        # ---------- 5) time-offset each burst -------------------------
        if self.max_shift > 0.0:
            max_samp = int(round(self.max_shift * self.sample_rate))
            if max_samp > 0:
                shifts = torch.randint(
                    -max_samp, max_samp + 1,        # inclusive
                    (cross_f.shape[0],),
                    device=device,
                )
                cross_f = self._roll_batch(cross_f, shifts)
                plus_f  = self._roll_batch(plus_f,  shifts)

        # ---------- 6) reshape back and sum ---------------------------
        N = cross_f.shape[-1]
        cross = torch.zeros(batch, self.n_max, N, dtype=dtype, device=device)
        plus  = torch.zeros_like(cross)

        cross[active] = cross_f           # boolean indexing keeps it 1-D
        plus [active] = plus_f

        cross = cross.sum(dim=1)          # (batch, N)
        plus  = plus .sum(dim=1)

        return cross, plus