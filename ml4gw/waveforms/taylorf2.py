import torch
from torchtyping import TensorType

GAMMA = 0.577215664901532860606512090082402431
"""Euler-Mascheroni constant. Same as lal.GAMMA"""

MSUN_SI = 1.988409870698050731911960804878414216e30
"""Solar mass in kg. Same as lal.MSUN_SI"""

MTSUN_SI = 4.925490947641266978197229498498379006e-6
"""1 solar mass in seconds. Same value as lal.MTSUN_SI"""

PI = 3.141592653589793238462643383279502884
"""Archimedes constant. Same as lal.PI"""

MPC_SEC = 1.02927125e14
"""
1 Mpc in seconds.
"""


def taylorf2_phase(
    f: TensorType,
    mass1: TensorType,
    mass2: TensorType,
) -> TensorType:
    """
    Calculate the inspiral phase for the IMRPhenomD waveform.
    """
    mass1_s = mass1 * MTSUN_SI
    mass2_s = mass2 * MTSUN_SI
    M_s = mass1_s + mass2_s
    eta = mass1_s * mass2_s / M_s / M_s

    Mf = (f.T * M_s).T

    v0 = torch.ones_like(Mf)
    v1 = (PI * Mf) ** (1.0 / 3.0)
    v2 = v1 * v1
    v3 = v2 * v1
    v4 = v3 * v1
    v5 = v4 * v1
    v6 = v5 * v1
    v7 = v6 * v1
    logv = torch.log(v1)
    v5_logv = v5 * logv
    v6_logv = v6 * logv

    # Phase coeffeciencts from https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiralPNCoefficients.c  # noqa E501
    pfaN = 3.0 / (128.0 * eta)
    pfa_v0 = 1.0
    pfa_v1 = 0.0
    pfa_v2 = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    pfa_v3 = -16.0 * PI
    pfa_v4 = (
        5.0
        * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta)
        / 72.0
    )
    pfa_v5logv = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    pfa_v5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    pfa_v6logv = -684.8 / 2.1
    pfa_v6 = (
        11583.231236531 / 4.694215680
        - 640.0 / 3.0 * PI * PI
        - 684.8 / 2.1 * GAMMA
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI * PI)
        + eta * eta * 76.055 / 1.728
        - eta * eta * eta * 127.825 / 1.296
        + pfa_v6logv * torch.log(torch.tensor(4.0))
    )
    pfa_v7 = PI * (
        770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta
    )
    # construct power series
    phasing = (v7.T * pfa_v7).T
    phasing += (v6.T * pfa_v6 + v6_logv.T * pfa_v6logv).T
    phasing += (v5.T * pfa_v5 + v5_logv.T * pfa_v5logv).T
    phasing += (v4.T * pfa_v4).T
    phasing += (v3.T * pfa_v3).T
    phasing += (v2.T * pfa_v2).T
    phasing += (v1.T * pfa_v1).T
    phasing += (v0.T * pfa_v0).T
    # Divide by 0PN v-dependence
    phasing /= v5
    # Multiply by 0PN coefficient
    phasing = (phasing.T * pfaN).T

    return phasing


def taylorf2_amplitude(f: TensorType, mass1, mass2, distance) -> TensorType:
    mass1_s = mass1 * MTSUN_SI
    mass2_s = mass2 * MTSUN_SI
    M_s = mass1_s + mass2_s
    eta = mass1_s * mass2_s / M_s / M_s
    Mf = (f.T * M_s).T
    v = (PI * Mf) ** (1.0 / 3.0)
    v10 = v**10

    # Flux and energy coefficient at newtonian
    FTaN = 32.0 * eta * eta / 5.0
    dETaN = 2 * (-eta / 2.0)

    amp0 = -4.0 * mass1_s * mass2_s * (PI / 12.0) ** 0.5

    amp0 /= distance * MPC_SEC
    flux = (v10.T * FTaN).T
    dEnergy = (v.T * dETaN).T
    amp = torch.sqrt(-dEnergy / flux) * v
    amp = (amp.T * amp0).T

    return amp


def taylorf2_htilde(
    f: TensorType,
    mass1: TensorType,
    mass2: TensorType,
    distance: TensorType,
    phic: TensorType,
    f_ref: float,
):
    # frequency array is repeated along batch
    f = f.repeat([mass1.shape[0], 1])
    f_ref = torch.tensor(f_ref).repeat([mass1.shape[0], 1])

    Psi = taylorf2_phase(f, mass1, mass2)
    Psi_ref = taylorf2_phase(f_ref, mass1, mass2)

    Psi = (Psi.T - 2 * phic).T
    Psi -= Psi_ref

    amp0 = taylorf2_amplitude(f, mass1, mass2, distance)
    h0 = amp0 * torch.exp(-1j * (Psi - PI / 4))
    return h0


def TaylorF2(
    f: TensorType,
    mass1: TensorType,
    mass2: TensorType,
    distance: TensorType,
    phic: TensorType,
    inclination: TensorType,
    f_ref: float,
):
    """
    TaylorF2 up to 3.5 PN in phase. Newtonian SPA amplitude.

    Returns:
    --------
      hp, hc
    """
    # shape assumed (n_batch, params)
    if (
        mass1.shape[0] != mass2.shape[0]
        or mass2.shape[0] != distance.shape[0]
        or distance.shape[0] != phic.shape[0]
        or phic.shape[0] != inclination.shape[0]
    ):
        raise RuntimeError("Tensors should have same batch size")
    cfac = torch.cos(inclination)
    pfac = 0.5 * (1.0 + cfac * cfac)

    htilde = taylorf2_htilde(f, mass1, mass2, distance, phic, f_ref)

    hp = (htilde.T * pfac).T
    hc = -1j * (htilde.T * cfac).T

    return hp, hc
