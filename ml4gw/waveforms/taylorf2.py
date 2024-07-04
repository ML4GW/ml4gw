import torch
from torchtyping import TensorType

from ..constants import MPC_SEC, MTSUN_SI, PI
from ..constants import EulerGamma as GAMMA


class TaylorF2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        f: TensorType,
        chirp_mass: TensorType,
        mass_ratio: TensorType,
        chi1: TensorType,
        chi2: TensorType,
        distance: TensorType,
        phic: TensorType,
        inclination: TensorType,
        f_ref: float,
    ):
        """
        TaylorF2 up to 3.5 PN in phase. Newtonian SPA amplitude.

        Args:
            f:
                Frequency series in Hz.
            chirp_mass:
                Chirp mass in solar masses
            mass_ratio:
                Mass ratio m1/m2
            chi1:
                Spin of m1
            chi2:
                Spin of m2
            distance:
                Luminosity distance
            phic:
                Phase at coalescence
            inclination:
                Inclination angle
            f_ref:
                Reference frequency

        Returns:
            hc, hp: Tuple[torch.Tensor, torch.Tensor]
                Cross and plus polarizations
        """

        # shape assumed (n_batch, params)
        if (
            chirp_mass.shape[0] != mass_ratio.shape[0]
            or chirp_mass.shape[0] != chi1.shape[0]
            or chi1.shape[0] != chi2.shape[0]
            or chi2.shape[0] != distance.shape[0]
            or distance.shape[0] != phic.shape[0]
            or phic.shape[0] != inclination.shape[0]
        ):
            raise RuntimeError("Tensors should have same batch size")
        mass2 = chirp_mass * (1.0 + mass_ratio) ** 0.2 / mass_ratio**0.6
        mass1 = mass_ratio * mass2
        cfac = torch.cos(inclination)
        pfac = 0.5 * (1.0 + cfac * cfac)

        htilde = self.taylorf2_htilde(
            f, mass1, mass2, chi1, chi2, distance, phic, f_ref
        )

        hp = (htilde.mT * pfac).mT
        hc = -1j * (htilde.mT * cfac).mT

        return hc, hp

    def taylorf2_htilde(
        self,
        f: TensorType,
        mass1: TensorType,
        mass2: TensorType,
        chi1: TensorType,
        chi2: TensorType,
        distance: TensorType,
        phic: TensorType,
        f_ref: float,
    ):
        mass1_s = mass1 * MTSUN_SI
        mass2_s = mass2 * MTSUN_SI
        M_s = mass1_s + mass2_s
        eta = mass1_s * mass2_s / M_s / M_s

        Mf = torch.outer(M_s, f)
        Mf_ref = torch.outer(M_s, f_ref * torch.ones_like(f))

        Psi, _ = self.taylorf2_phase(Mf, mass1, mass2, chi1, chi2)
        Psi_ref, _ = self.taylorf2_phase(Mf_ref, mass1, mass2, chi1, chi2)

        Psi = (Psi.mT - 2 * phic).mT
        Psi -= Psi_ref

        amp0 = self.taylorf2_amplitude(Mf, mass1, mass2, eta, distance)
        h0 = amp0 * torch.exp(-1j * (Psi - PI / 4))
        return h0

    def taylorf2_amplitude(
        self, Mf: TensorType, mass1, mass2, eta, distance
    ) -> TensorType:
        mass1_s = mass1 * MTSUN_SI
        mass2_s = mass2 * MTSUN_SI
        v = (PI * Mf) ** (1.0 / 3.0)
        v10 = v**10

        # Flux and energy coefficient at newtonian
        FTaN = 32.0 * eta * eta / 5.0
        dETaN = 2 * (-eta / 2.0)

        amp0 = -4.0 * mass1_s * mass2_s * (PI / 12.0) ** 0.5

        amp0 /= distance * MPC_SEC
        flux = (v10.mT * FTaN).mT
        dEnergy = (v.mT * dETaN).mT
        amp = torch.sqrt(-dEnergy / flux) * v
        amp = (amp.mT * amp0).mT

        return amp

    def taylorf2_phase(
        self,
        Mf: TensorType,
        mass1: TensorType,
        mass2: TensorType,
        chi1: TensorType,
        chi2: TensorType,
    ) -> TensorType:
        """
        Calculate the inspiral phase for the TaylorF2.
        """
        M = mass1 + mass2
        eta = mass1 * mass2 / M / M
        m1byM = mass1 / M
        m2byM = mass2 / M
        chi1sq = chi1 * chi1
        chi2sq = chi2 * chi2

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
        # SO contributions at 1.5 PN
        pfa_v3 += (
            m1byM * (25.0 + 38.0 / 3.0 * m1byM) * chi1
            + m2byM * (25.0 + 38.0 / 3.0 * m2byM) * chi2
        )
        pfa_v4 = (
            5.0
            * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta)
            / 72.0
        )
        # SO, SS, S1,2-squared contributions
        pfa_v4 += (
            247.0 / 4.8 * eta * chi1 * chi2
            + -721.0 / 4.8 * eta * chi1 * chi2
            + (-720.0 / 9.6 * m1byM * m1byM + 1.0 / 9.6 * m1byM * m1byM)
            * chi1sq
            + (-720.0 / 9.6 * m2byM * m2byM + 1.0 / 9.6 * m2byM * m2byM)
            * chi2sq
            + (240.0 / 9.6 * m1byM * m1byM + -7.0 / 9.6 * m1byM * m1byM)
            * chi1sq
            + (240.0 / 9.6 * m2byM * m2byM + -7.0 / 9.6 * m2byM * m2byM)
            * chi2sq
        )
        pfa_v5logv = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * PI
        pfa_v5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI
        # SO coefficient for 2.5 PN
        pfa_v5logv += 3.0 * (
            -m1byM
            * (
                1391.5 / 8.4
                - 10.0 / 3.0 * m1byM * (1.0 - m1byM)
                + m1byM * (1276.0 / 8.1 + 170.0 / 9.0 * m1byM * (1.0 - m1byM))
            )
            * chi1
            - m2byM
            * (
                1391.5 / 8.4
                - 10.0 / 3.0 * m2byM * (1.0 - m2byM)
                + m2byM * (1276.0 / 8.1 + 170.0 / 9.0 * m2byM * (1.0 - m2byM))
            )
            * chi2
        )
        pfa_v5 += (
            -m1byM
            * (
                1391.5 / 8.4
                - 10.0 / 3.0 * m1byM * (1.0 - m1byM)
                + m1byM * (1276.0 / 8.1 + 170.0 / 9.0 * m1byM * (1.0 - m1byM))
            )
            * chi1
            + -m2byM
            * (
                1391.5 / 8.4
                - 10.0 / 3.0 * m2byM * (1.0 - m2byM)
                + m2byM * (1276.0 / 8.1 + 170.0 / 9.0 * m2byM * (1.0 - m2byM))
            )
            * chi2
        )
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
        # SO + S1-S2 + S-squared contribution at 3 PN
        pfa_v6 += (
            PI * m1byM * (1490.0 / 3.0 + m1byM * 260.0) * chi1
            + PI * m2byM * (1490.0 / 3.0 + m2byM * 260.0) * chi2
            + (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2
            + (
                (4703.5 / 8.4 + 2935.0 / 6.0 * m1byM - 120.0 * m1byM * m1byM)
                * m1byM
                * m1byM
                + (
                    -4108.25 / 6.72
                    - 108.5 / 1.2 * m1byM
                    + 125.5 / 3.6 * m1byM * m1byM
                )
                * m1byM
                * m1byM
            )
            * chi1sq
            + (
                (4703.5 / 8.4 + 2935.0 / 6.0 * m2byM - 120.0 * m2byM * m2byM)
                * m2byM
                * m2byM
                + (
                    -4108.25 / 6.72
                    - 108.5 / 1.2 * m2byM
                    + 125.5 / 3.6 * m2byM * m2byM
                )
                * m2byM
                * m2byM
            )
            * chi2sq
        )
        pfa_v7 = PI * (
            770.96675 / 2.54016
            + 378.515 / 1.512 * eta
            - 740.45 / 7.56 * eta * eta
        )
        # SO contribution at 3.5 PN
        pfa_v7 += (
            m1byM
            * (
                -17097.8035 / 4.8384
                + eta * 28764.25 / 6.72
                + eta * eta * 47.35 / 1.44
                + m1byM
                * (
                    -7189.233785 / 1.524096
                    + eta * 458.555 / 3.024
                    - eta * eta * 534.5 / 7.2
                )
            )
        ) * chi1 + (
            m2byM
            * (
                -17097.8035 / 4.8384
                + eta * 28764.25 / 6.72
                + eta * eta * 47.35 / 1.44
                + m2byM
                * (
                    -7189.233785 / 1.524096
                    + eta * 458.555 / 3.024
                    - eta * eta * 534.5 / 7.2
                )
            )
        ) * chi2
        # construct power series
        phasing = (v7.mT * pfa_v7).mT
        phasing += (v6.mT * pfa_v6 + v6_logv.mT * pfa_v6logv).mT
        phasing += (v5.mT * pfa_v5 + v5_logv.mT * pfa_v5logv).mT
        phasing += (v4.mT * pfa_v4).mT
        phasing += (v3.mT * pfa_v3).mT
        phasing += (v2.mT * pfa_v2).mT
        phasing += (v1.mT * pfa_v1).mT
        phasing += (v0.mT * pfa_v0).mT
        # Divide by 0PN v-dependence
        phasing /= v5
        # Multiply by 0PN coefficient
        phasing = (phasing.mT * pfaN).mT

        # Derivative of phase w.r.t Mf
        # dPhi/dMf = dPhi/dv dv/dMf
        Dphasing = (2.0 * v7.mT * pfa_v7).mT
        Dphasing += (v6.mT * (pfa_v6 + pfa_v6logv)).mT
        Dphasing += (v6_logv.mT * pfa_v6logv).mT
        Dphasing += (v5.mT * pfa_v5logv).mT
        Dphasing += (-1.0 * v4.mT * pfa_v4).mT
        Dphasing += (-2.0 * v3.mT * pfa_v3).mT
        Dphasing += (-3.0 * v2.mT * pfa_v2).mT
        Dphasing += (-4.0 * v1.mT * pfa_v1).mT
        Dphasing += -5.0 * v0
        Dphasing /= 3.0 * v1 * v7
        Dphasing *= PI
        Dphasing = (Dphasing.mT * pfaN).mT

        return phasing, Dphasing
