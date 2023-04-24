import torch
from scipy.signal.windows import tukey
from torch import Tensor

from ml4gw.types import ScalarTensor


def semi_major_minor_from_e(e: Tensor):
    a = 1.0 / torch.sqrt(2.0 - e * e)
    b = a * torch.sqrt(1.0 - e * e)
    return a, b


# TODO: replace with torch implementation
def tukey_window(num: int, alpha: float = 0.5):
    return torch.tensor(tukey(num, alpha=alpha))


def sine_gaussian(
    quality: ScalarTensor,
    frequency: ScalarTensor,
    hrss: ScalarTensor,
    phase: ScalarTensor,
    eccentricity: ScalarTensor,
    sample_rate: float,
    duration: float,
):
    """
    Generate lalsimulation implementation of a sine-Gaussian waveform.
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimBurst.c#L1080

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
        sample_rate:
            Sample rate of the waveform
        duration:
            Duration of the waveform

    Returns:
        A tensor of size (batch, 2, time) containing
        the hplus and hcross polarizations
    """

    # add dimension for calculating waveforms in batch
    frequency = frequency.view(-1, 1)
    quality = quality.view(-1, 1)
    hrss = hrss.view(-1, 1)
    phase = phase.view(-1, 1)
    eccentricity = eccentricity.view(-1, 1)

    # determine times based on requested duration and sample rate
    # and shift so that the waveform is centered at t=0
    num = int(duration * sample_rate)
    times = torch.arange(num, dtype=torch.float64) / sample_rate
    times -= duration / 2.0

    # calculate relative hplus / hcross amplitudes based on eccentricity
    # as well as normalization factors
    a, b = semi_major_minor_from_e(eccentricity)

    norm_prefactor = quality / (
        4.0 * frequency * torch.sqrt(torch.Tensor([torch.pi]))
    )
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
    phi = 2 * torch.pi * frequency * times
    complex_phase = torch.complex(torch.zeros_like(phi), (phi - phase))

    tukey = tukey_window(num)

    # calculate the waveform and apply a tukey window to taper the waveform
    fac = torch.exp(phi**2 / (-2.0 * quality**2) + complex_phase)
    fac *= tukey
    hplus = fac.real * h0_plus
    hcross = fac.imag * h0_cross

    hplus = hplus.unsqueeze(1)
    hcross = hcross.unsqueeze(1)

    waveforms = torch.cat([hplus, hcross], dim=1)
    return waveforms
