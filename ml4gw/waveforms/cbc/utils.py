"""
Utilities for conditioning waveforms
See https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c # noqa
"""
import torch

from ml4gw.constants import MRSUN, MSUN, MTSUN_SI, C, G
from ml4gw.types import BatchTensor
from ml4gw.waveforms.cbc import coefficients


def chirp_time_bound(
    fstart: float,
    mass_1: BatchTensor,
    mass_2: BatchTensor,
    s1: BatchTensor,
    s2: BatchTensor,
) -> BatchTensor:
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L4969
    """

    total_mass = mass_1 + mass_2
    reduced_mass = mass_1 * mass_2 / total_mass
    eta = reduced_mass / total_mass
    chi = torch.max(s1.abs(), s2.abs()).abs()

    c0 = torch.abs(coefficients.taylor_t2_timing_0pn_coeff(total_mass, eta))

    c2 = coefficients.taylor_t2_timing_2pn_coeff(eta)
    c3 = (226.0 / 15.0) * chi
    c4 = coefficients.taylor_t2_timing_4pn_coeff(eta)

    v = (torch.pi * total_mass * fstart * G) ** (1.0 / 3.0)
    v /= C

    return (c0 * (v**-8) * (1 + (c2 + (c3 + c4 * v) * v) * v * v)).float()


def chirp_start_frequency_bound(
    tchirp: BatchTensor,
    mass_1: BatchTensor,
    mass_2: BatchTensor,
):
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5104
    """
    total_mass = mass_1 + mass_2
    mu = mass_1 * mass_2 / total_mass

    eta = mu / total_mass
    c0 = coefficients.taylor_t3_frequency_0pn_coeff(total_mass)
    return (
        c0
        * pow(5.0 * total_mass * (MTSUN_SI / MSUN) / (eta * tchirp), 3.0 / 8.0)
    ).float()


def final_black_hole_spin_bound(
    s1z: BatchTensor, s2z: BatchTensor
) -> BatchTensor:
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5081
    """
    maximum_black_hole_spin = 0.998
    s = 0.686 + 0.15 * (s1z + s2z)
    s = torch.maximum(s, torch.abs(s1z)).maximum(torch.abs(s2z))
    s = torch.clamp(s, max=maximum_black_hole_spin)
    return s


def merge_time_bound(mass_1: BatchTensor, mass_2: BatchTensor) -> BatchTensor:
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5007
    """

    n_orbits = 1
    total_mass = mass_1 + mass_2
    r = 9.0 * total_mass * MRSUN / MSUN
    v = C / 3.0
    return (n_orbits * (2.0 * torch.pi * r / v)).float()


def ringdown_time_bound(
    total_mass: BatchTensor, s: BatchTensor
) -> BatchTensor:
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5032
    """
    n_efolds = 11

    f1 = 1.5251
    f2 = -1.1568
    f3 = 0.1292
    q1 = 0.7000
    q2 = 1.4187
    q3 = -0.4990

    omega = (f1 + f2 * (1.0 - s) ** f3) / (total_mass * MTSUN_SI / MSUN)
    Q = q1 + q2 * (1.0 - s) ** q3
    tau = 2.0 * Q / omega
    return (n_efolds * tau).float()


def frequency_isco(mass_1: BatchTensor, mass_2: BatchTensor):
    mass_1 = (mass_1 * MSUN).double()
    mass_2 = (mass_2 * MSUN).double()
    return (
        1.0 / 9.0**1.5 * torch.pi * (mass_1 + mass_2) * MTSUN_SI / MSUN
    ).float()
