import torch

from ml4gw.constants import C, G
from ml4gw.types import BatchTensor


def taylor_t2_timing_0pn_coeff(total_mass: BatchTensor, eta: BatchTensor):
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralPNCoefficients.c#L1528
    """

    output = total_mass * G / C**3
    return -5.0 * output / (256.0 * eta)


def taylor_t2_timing_2pn_coeff(eta: BatchTensor):
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralPNCoefficients.c#L1545
    """
    return 7.43 / 2.52 + 11.0 / 3.0 * eta


def taylor_t2_timing_4pn_coeff(eta: BatchTensor):
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralPNCoefficients.c#L1560
    """
    return 30.58673 / 5.08032 + 54.29 / 5.04 * eta + 61.7 / 7.2 * eta**2


def taylor_t3_frequency_0pn_coeff(total_mass: BatchTensor):
    """
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralPNCoefficients.c#L1723
    """
    output = total_mass * G / C**3.0
    return 1.0 / (8.0 * torch.pi * output)
