import torch
from jaxtyping import Float

from ml4gw.constants import MTSUN_SI, PI
from ml4gw.types import BatchTensor, FrequencySeries1d

from .phenom_d_data import QNMData_a, QNMData_fdamp, QNMData_fring
from .taylorf2 import TaylorF2


class IMRPhenomD(TaylorF2):
    def __init__(self):
        super().__init__()
        self.register_buffer("qnmdata_a", QNMData_a)
        self.register_buffer("qnmdata_fdamp", QNMData_fdamp)
        self.register_buffer("qnmdata_fring", QNMData_fring)

    def forward(
        self,
        f: FrequencySeries1d,
        chirp_mass: BatchTensor,
        mass_ratio: BatchTensor,
        chi1: BatchTensor,
        chi2: BatchTensor,
        distance: BatchTensor,
        phic: BatchTensor,
        inclination: BatchTensor,
        f_ref: float,
    ):
        """
        IMRPhenomD waveform

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
                Distance to source in Mpc
            phic:
                Phase at coalescence
            inclination:
                Inclination of the source
            f_ref:
                Reference frequency

        Returns:
            hc, hp: Tuple[torch.Tensor, torch.Tensor]
                Cross and plus polarizations
        """
        # shape assumed (n_batch, params)
        if (
            chirp_mass.shape[0] != mass_ratio.shape[0]
            or mass_ratio.shape[0] != chi1.shape[0]
            or chi1.shape[0] != chi2.shape[0]
            or chi2.shape[0] != distance.shape[0]
            or distance.shape[0] != phic.shape[0]
            or phic.shape[0] != inclination.shape[0]
        ):
            raise RuntimeError("Tensors should have same batch size")
        cfac = torch.cos(inclination)
        pfac = 0.5 * (1.0 + cfac * cfac)

        htilde = self.phenom_d_htilde(
            f, chirp_mass, mass_ratio, chi1, chi2, distance, phic, f_ref
        )

        hp = (htilde.mT * pfac).mT
        hc = -1j * (htilde.mT * cfac).mT

        return hc, hp

    def phenom_d_htilde(
        self,
        f: FrequencySeries1d,
        chirp_mass: BatchTensor,
        mass_ratio: BatchTensor,
        chi1: BatchTensor,
        chi2: BatchTensor,
        distance: BatchTensor,
        phic: BatchTensor,
        f_ref: float,
    ) -> Float[FrequencySeries1d, " batch"]:
        total_mass = chirp_mass * (1 + mass_ratio) ** 1.2 / mass_ratio**0.6
        mass_1 = total_mass / (1 + mass_ratio)
        mass_2 = mass_1 * mass_ratio
        eta = (chirp_mass / total_mass) ** (5 / 3)
        eta2 = eta * eta
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        chi = self.chiPN(Seta, eta, chi1, chi2)
        chi22 = chi2 * chi2
        chi12 = chi1 * chi1
        xi = -1.0 + chi
        M_s = total_mass * MTSUN_SI

        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)

        fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)
        Mf_peak = self.fmaxCalc(fRD, fDM, gamma2, gamma3)
        _, t0 = self.phenom_d_mrd_phase(Mf_peak, eta, eta2, chi1, chi2, xi)

        Mf = torch.outer(M_s, f)
        Mf_ref = torch.outer(M_s, f_ref * torch.ones_like(f))

        Psi, _ = self.phenom_d_phase(
            Mf, mass_1, mass_2, eta, eta2, chi1, chi2, xi
        )
        Psi_ref, _ = self.phenom_d_phase(
            Mf_ref, mass_1, mass_2, eta, eta2, chi1, chi2, xi
        )

        Psi = (Psi.mT - 2 * phic).mT
        Psi -= Psi_ref
        Psi -= ((Mf - Mf_ref).mT * t0).mT

        amp, _ = self.phenom_d_amp(
            Mf,
            mass_1,
            mass_2,
            eta,
            eta2,
            Seta,
            chi1,
            chi2,
            chi12,
            chi22,
            xi,
            distance,
        )

        amp_0 = self.taylorf2_amplitude(
            Mf, mass_1, mass_2, eta, distance
        )  # this includes f^(-7/6) dependence

        h0 = -amp_0 * amp * torch.exp(-1j * Psi)

        return h0

    def phenom_d_amp(
        self,
        Mf,
        mass_1,
        mass_2,
        eta,
        eta2,
        Seta,
        chi1,
        chi2,
        chi12,
        chi22,
        xi,
        distance,
        fRD=None,  # used for passing ringdown frequency from phenom_p
        fDM=None,  # used for passing damping frequency from phenom_p
    ):
        ins_amp, ins_Damp = self.phenom_d_inspiral_amp(
            Mf, eta, eta2, Seta, xi, chi1, chi2, chi12, chi22
        )
        int_amp, int_Damp = self.phenom_d_int_amp(
            Mf, eta, eta2, Seta, chi1, chi2, chi12, chi22, xi, fRD, fDM
        )
        mrd_amp, mrd_Damp = self.phenom_d_mrd_amp(
            Mf, eta, eta2, chi1, chi2, xi, fRD, fDM
        )

        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)

        # merger ringdown
        if (fRD is None) or (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)
        Mf_peak = self.fmaxCalc(fRD, fDM, gamma2, gamma3)

        # Geometric peak and joining frequencies
        Mf_peak = (torch.ones_like(Mf).mT * Mf_peak).mT
        Mf_join_ins = 0.014 * torch.ones_like(Mf)

        # construct full IMR Amp
        theta_minus_f1 = torch.heaviside(
            Mf_join_ins - Mf, torch.tensor(0.0, device=Mf.device)
        )
        theta_plus_f1 = torch.heaviside(
            Mf - Mf_join_ins, torch.tensor(1.0, device=Mf.device)
        )
        theta_minus_f2 = torch.heaviside(
            Mf_peak - Mf, torch.tensor(0.0, device=Mf.device)
        )
        theta_plus_f2 = torch.heaviside(
            Mf - Mf_peak, torch.tensor(1.0, device=Mf.device)
        )

        amp = theta_minus_f1 * ins_amp
        amp += theta_plus_f1 * int_amp * theta_minus_f2
        amp += theta_plus_f2 * mrd_amp

        Damp = theta_minus_f1 * ins_Damp
        Damp += theta_plus_f1 * int_Damp * theta_minus_f2
        Damp += theta_plus_f2 * mrd_Damp

        return amp, Damp

    def phenom_d_int_amp(
        self,
        Mf,
        eta,
        eta2,
        Seta,
        chi1,
        chi2,
        chi12,
        chi22,
        xi,
        fRD=None,  # used for passing ringdown frequency from phenom_p
        fDM=None,  # used for passing damping frequency from phenom_p
    ):
        # merger ringdown
        if (fRD is None) or (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)
        # Geometric frequency definition from PhenomD header file
        AMP_fJoin_INS = 0.014

        Mf1 = AMP_fJoin_INS * torch.ones_like(Mf)
        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)

        fpeak = self.fmaxCalc(fRD, fDM, gamma2, gamma3)
        Mf3 = (torch.ones_like(Mf).mT * fpeak).mT
        dfx = 0.5 * (Mf3 - Mf1)
        Mf2 = Mf1 + dfx

        v1, d1 = self.phenom_d_inspiral_amp(
            Mf1, eta, eta2, Seta, xi, chi1, chi2, chi12, chi22
        )
        v3, d2 = self.phenom_d_mrd_amp(
            Mf3, eta, eta2, chi1, chi2, xi, fRD, fDM
        )
        v2 = (
            torch.ones_like(Mf).mT * self.AmpIntColFitCoeff(eta, eta2, xi)
        ).mT

        delta_0, delta_1, delta_2, delta_3, delta_4 = self.delta_values(
            f1=Mf1, f2=Mf2, f3=Mf3, v1=v1, v2=v2, v3=v3, d1=d1, d2=d2
        )

        amp = (
            delta_0
            + Mf * delta_1
            + Mf**2 * (delta_2 + Mf * delta_3 + Mf**2 * delta_4)
        )
        Damp = delta_1 + Mf * (
            2 * delta_2 + 3 * Mf * delta_3 + 4 * Mf**2 * delta_4
        )
        return amp, Damp

    def phenom_d_mrd_amp(
        self, Mf, eta, eta2, chi1, chi2, xi, fRD=None, fDM=None
    ):
        # merger ringdown
        if (fRD is None) or (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)

        gamma1 = self.gamma1_fun(eta, eta2, xi)
        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)
        fDMgamma3 = fDM * gamma3
        pow2_fDMgamma3 = (torch.ones_like(Mf).mT * fDMgamma3 * fDMgamma3).mT
        fminfRD = Mf - (torch.ones_like(Mf).mT * fRD).mT
        exp_times_lorentzian = torch.exp(fminfRD.mT * gamma2 / fDMgamma3).mT
        exp_times_lorentzian *= fminfRD**2 + pow2_fDMgamma3

        amp = (1 / exp_times_lorentzian.mT * gamma1 * gamma3 * fDM).mT
        Damp = (fminfRD.mT * -2 * fDM * gamma1 * gamma3) / (
            fminfRD * fminfRD + pow2_fDMgamma3
        ).mT - (gamma2 * gamma1)
        Damp = Damp.mT / exp_times_lorentzian
        return amp, Damp

    def phenom_d_inspiral_amp(
        self, Mf, eta, eta2, Seta, xi, chi1, chi2, chi12, chi22
    ):
        SetaPlus1 = 1 + Seta

        Mf_one_third = Mf ** (1.0 / 3.0)
        Mf_two_third = Mf_one_third * Mf_one_third
        Mf_four_third = Mf_two_third * Mf_two_third
        Mf_five_third = Mf_four_third * Mf_one_third
        Mf_seven_third = Mf_five_third * Mf_two_third
        MF_eight_third = Mf_seven_third * Mf_one_third
        Mf_two = Mf * Mf
        Mf_three = Mf_two * Mf

        prefactors_two_thirds = ((-969 + 1804 * eta) * PI ** (2.0 / 3.0)) / 672
        prefactors_one = (
            (
                chi1 * (81 * SetaPlus1 - 44 * eta)
                + chi2 * (81 - 81 * Seta - 44 * eta)
            )
            * PI
        ) / 48.0
        prefactors_four_thirds = (
            (
                -27312085.0
                - 10287648 * chi22
                - 10287648 * chi12 * SetaPlus1
                + 10287648 * chi22 * Seta
                + 24
                * (
                    -1975055
                    + 857304 * chi12
                    - 994896 * chi1 * chi2
                    + 857304 * chi22
                )
                * eta
                + 35371056 * eta2
            )
            * PI ** (4.0 / 3.0)
        ) / 8.128512e6
        prefactors_five_thirds = (
            PI ** (5.0 / 3.0)
            * (
                chi2
                * (
                    -285197 * (-1 + Seta)
                    + 4 * (-91902 + 1579 * Seta) * eta
                    - 35632 * eta2
                )
                + chi1
                * (
                    285197 * SetaPlus1
                    - 4 * (91902 + 1579 * Seta) * eta
                    - 35632 * eta2
                )
                + 42840 * (-1.0 + 4 * eta) * PI
            )
        ) / 32256.0
        prefactors_two = (
            -(
                PI**2
                * (
                    -336
                    * (
                        -3248849057.0
                        + 2943675504 * chi12
                        - 3339284256 * chi1 * chi2
                        + 2943675504 * chi22
                    )
                    * eta2
                    - 324322727232 * eta2 * eta
                    - 7
                    * (
                        -177520268561
                        + 107414046432 * chi22
                        + 107414046432 * chi12 * SetaPlus1
                        - 107414046432 * chi22 * Seta
                        + 11087290368
                        * (chi1 + chi2 + chi1 * Seta - chi2 * Seta)
                        * PI
                    )
                    + 12
                    * eta
                    * (
                        -545384828789
                        - 176491177632 * chi1 * chi2
                        + 202603761360 * chi22
                        + 77616 * chi12 * (2610335 + 995766 * Seta)
                        - 77287373856 * chi22 * Seta
                        + 5841690624 * (chi1 + chi2) * PI
                        + 21384760320 * PI**2
                    )
                )
            )
            / 6.0085960704e10
        )
        prefactors_seven_thirds = self.rho1_fun(eta, eta2, xi)
        prefactors_eight_thirds = self.rho2_fun(eta, eta2, xi)
        prefactors_three = self.rho3_fun(eta, eta2, xi)

        amp = torch.ones_like(Mf)
        amp += (
            Mf_two_third.mT * prefactors_two_thirds
            + Mf_four_third.mT * prefactors_four_thirds
            + Mf_five_third.mT * prefactors_five_thirds
            + Mf_seven_third.mT * prefactors_seven_thirds
            + MF_eight_third.mT * prefactors_eight_thirds
            + Mf.mT * prefactors_one
            + Mf_two.mT * prefactors_two
            + Mf_three.mT * prefactors_three
        ).mT

        Damp = (
            (2.0 / 3.0) / Mf_one_third.mT * prefactors_two_thirds
            + (4.0 / 3.0) * Mf_one_third.mT * prefactors_four_thirds
            + (5.0 / 3.0) * Mf_two_third.mT * prefactors_five_thirds
            + (7.0 / 3.0) * Mf_four_third.mT * prefactors_seven_thirds
            + (8.0 / 3.0) * Mf_five_third.mT * prefactors_eight_thirds
            + prefactors_one
            + 2.0 * Mf.mT * prefactors_two
            + 3.0 * Mf_two.mT * prefactors_three
        ).mT

        return amp, Damp

    def phenom_d_phase(
        self, Mf, mass_1, mass_2, eta, eta2, chi1, chi2, xi, fRD=None, fDM=None
    ):
        ins_phase, ins_Dphase = self.phenom_d_inspiral_phase(
            Mf, mass_1, mass_2, eta, eta2, xi, chi1, chi2
        )
        int_phase, int_Dphase = self.phenom_d_int_phase(Mf, eta, eta2, xi)
        mrd_phase, mrd_Dphase = self.phenom_d_mrd_phase(
            Mf, eta, eta2, chi1, chi2, xi, fRD, fDM
        )

        # merger ringdown
        if (fRD is None) or (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)
        # definitions in Eq. (35) of arXiv:1508.07253
        # PHI_fJoin_INS in header LALSimIMRPhenomD.h
        # C1 continuity at intermediate region i.e. f_1
        PHI_fJoin_INS = 0.018 * torch.ones_like(Mf)
        ins_phase_f1, ins_Dphase_f1 = self.phenom_d_inspiral_phase(
            PHI_fJoin_INS, mass_1, mass_2, eta, eta2, xi, chi1, chi2
        )
        int_phase_f1, int_Dphase_f1 = self.phenom_d_int_phase(
            PHI_fJoin_INS, eta, eta2, xi
        )
        C2Int = ins_Dphase_f1 - int_Dphase_f1
        C1Int = (
            ins_phase_f1 - (int_phase_f1.mT / eta).mT - C2Int * PHI_fJoin_INS
        )
        # C1 continuity at ringdown
        fRDJoin = (0.5 * torch.ones_like(Mf).mT * fRD).mT
        int_phase_rd, int_Dphase_rd = self.phenom_d_int_phase(
            fRDJoin, eta, eta2, xi
        )
        mrd_phase_rd, mrd_Dphase_rd = self.phenom_d_mrd_phase(
            fRDJoin, eta, eta2, chi1, chi2, xi, fRD, fDM
        )
        PhiIntTempVal = (int_phase_rd.mT / eta).mT + C1Int + C2Int * fRDJoin
        # C2MRD = int_Dphase_rd - mrd_Dphase_rd
        C2MRD = C2Int + int_Dphase_rd - mrd_Dphase_rd
        C1MRD = PhiIntTempVal - (mrd_phase_rd.mT / eta).mT - C2MRD * fRDJoin

        int_phase = (int_phase.mT / eta).mT
        int_phase += C1Int
        int_phase += Mf * C2Int

        mrd_phase = (mrd_phase.mT / eta).mT
        mrd_phase += C1MRD
        mrd_phase += Mf * C2MRD

        # construct full IMR phase
        theta_minus_f1 = torch.heaviside(
            PHI_fJoin_INS - Mf, torch.tensor(0.0, device=Mf.device)
        )
        theta_plus_f1 = torch.heaviside(
            Mf - PHI_fJoin_INS, torch.tensor(1.0, device=Mf.device)
        )
        theta_minus_f2 = torch.heaviside(
            fRDJoin - Mf, torch.tensor(0.0, device=Mf.device)
        )
        theta_plus_f2 = torch.heaviside(
            Mf - fRDJoin, torch.tensor(1.0, device=Mf.device)
        )

        phasing = theta_minus_f1 * ins_phase
        phasing += theta_plus_f1 * int_phase * theta_minus_f2
        phasing += theta_plus_f2 * mrd_phase

        Dphasing = theta_minus_f1 * ins_Dphase
        Dphasing += theta_plus_f1 * int_Dphase * theta_minus_f2
        Dphasing += theta_plus_f2 * mrd_Dphase

        return phasing, Dphasing

    def phenom_d_mrd_phase(
        self, Mf, eta, eta2, chi1, chi2, xi, fRD=None, fDM=None
    ):
        alpha1 = self.alpha1Fit(eta, eta2, xi)
        alpha2 = self.alpha2Fit(eta, eta2, xi)
        alpha3 = self.alpha3Fit(eta, eta2, xi)
        alpha4 = self.alpha4Fit(eta, eta2, xi)
        alpha5 = self.alpha5Fit(eta, eta2, xi)

        # merger ringdown
        if (fRD is None) or (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)
        f_minus_alpha5_fRD = (Mf.t() - alpha5 * fRD).t()

        # Leading 1/eta is not multiplied at this stage
        mrd_phasing = (Mf.t() * alpha1).t()
        mrd_phasing -= (1 / Mf.t() * alpha2).t()
        mrd_phasing += (4.0 / 3.0) * (Mf.t() ** (3.0 / 4.0) * alpha3).t()
        mrd_phasing += (torch.atan(f_minus_alpha5_fRD.t() / fDM) * alpha4).t()

        mrd_Dphasing = (
            alpha4 * fDM / (f_minus_alpha5_fRD.t() ** 2 + fDM**2)
        ).t()
        mrd_Dphasing += (Mf.t() ** (-1.0 / 4.0) * alpha3).t()
        mrd_Dphasing += (Mf.t() ** (-2.0) * alpha2).t()
        mrd_Dphasing = (mrd_Dphasing.t() + alpha1).t()
        mrd_Dphasing = (mrd_Dphasing.t() / eta).t()

        return mrd_phasing, mrd_Dphasing

    def phenom_d_int_phase(self, Mf, eta, eta2, xi):
        beta1 = self.beta1Fit(eta, eta2, xi)
        beta2 = self.beta2Fit(eta, eta2, xi)
        beta3 = self.beta3Fit(eta, eta2, xi)
        # Merger phase
        # Leading beta0 is not added here
        # overall 1/eta is not multiplied
        int_phasing = (Mf.mT * beta1).mT
        int_phasing += (torch.log(Mf).mT * beta2).mT
        int_phasing -= (Mf.mT ** (-3.0) / 3.0 * beta3).mT

        # overall 1/eta is multiple in derivative of
        # intermediate phase
        int_Dphasing = (Mf.mT ** (-4.0) * beta3).mT
        int_Dphasing += (Mf.mT ** (-1.0) * beta2).mT
        int_Dphasing = (int_Dphasing.mT + beta1).mT
        int_Dphasing = (int_Dphasing.mT / eta).mT
        return int_phasing, int_Dphasing

    def subtract3PNSS(self, Mf, mass1, mass2, eta, eta2, xi, chi1, chi2):
        M = mass1 + mass2
        eta = mass1 * mass2 / M / M
        m1byM = mass1 / M
        m2byM = mass2 / M
        chi1sq = chi1 * chi1
        chi2sq = chi2 * chi2
        v1 = (PI * Mf) ** (1.0 / 3.0)
        v5 = v1**5.0
        v6 = v1**6.0
        v7 = v1**7.0
        pfaN = 3.0 / (128.0 * eta)
        pn_ss3 = (
            (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2
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
        phase_pn_ss3 = (((v6.mT * pn_ss3).mT / v5).mT * pfaN).mT
        Dphase_pn_ss3 = (
            (PI * (v6.mT * pn_ss3).mT / (3.0 * v1 * v7)).mT * pfaN
        ).mT
        return phase_pn_ss3, Dphase_pn_ss3

    def phenom_d_inspiral_phase(
        self, Mf, mass_1, mass_2, eta, eta2, xi, chi1, chi2
    ):
        ins_phasing, ins_Dphasing = self.taylorf2_phase(
            Mf, mass_1, mass_2, chi1, chi2
        )
        # subtract 3PN spin-spin term as this is in LAL's TaylorF2
        # implementation, but was not available when PhenomD was tuned.
        # refer https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomD.c#L397-398 # noqa: E501
        pn_ss3, Dpn_ss3 = self.subtract3PNSS(
            Mf, mass_1, mass_2, eta, eta2, xi, chi1, chi2
        )
        ins_phasing -= pn_ss3
        ins_Dphasing -= Dpn_ss3

        sigma1 = self.sigma1Fit(eta, eta2, xi)
        sigma2 = self.sigma2Fit(eta, eta2, xi)
        sigma3 = self.sigma3Fit(eta, eta2, xi)
        sigma4 = self.sigma4Fit(eta, eta2, xi)

        ins_phasing += (Mf.mT * sigma1 / eta).mT
        ins_phasing += (Mf.mT ** (4.0 / 3.0) * 0.75 * sigma2 / eta).mT
        ins_phasing += (Mf.mT ** (5.0 / 3.0) * 0.6 * sigma3 / eta).mT
        ins_phasing += (Mf.mT**2.0 * 0.5 * sigma4 / eta).mT

        ins_Dphasing = (ins_Dphasing.mT + sigma1 / eta).mT
        ins_Dphasing += (Mf.mT ** (1.0 / 3.0) * sigma2 / eta).mT
        ins_Dphasing += (Mf.mT ** (2.0 / 3.0) * sigma3 / eta).mT
        ins_Dphasing += (Mf.mT * sigma4 / eta).mT

        return ins_phasing, ins_Dphasing

    def fring_fdamp(self, eta, eta2, chi1, chi2):
        finspin = self.FinalSpin0815(eta, eta2, chi1, chi2)
        Erad = self.PhenomInternal_EradRational0815(eta, eta2, chi1, chi2)

        fRD, fDM = self._linear_interp_finspin(finspin)
        fRD /= 1.0 - Erad
        fDM /= 1.0 - Erad

        return fRD, fDM

    def fmaxCalc(self, fRD, fDM, gamma2, gamma3):
        mask = gamma2 <= 1
        # calculate result for gamma2 <= 1 case
        sqrt_term = torch.sqrt(1 - gamma2.pow(2))
        result_case1 = fRD + (fDM * (-1 + sqrt_term) * gamma3) / gamma2

        # calculate result for gamma2 > 1 case
        # i.e. don't add sqrt term
        result_case2 = fRD + (-fDM * gamma3) / gamma2

        # combine results using mask
        result = torch.where(mask, result_case1, result_case2)

        return torch.abs(result)

    def _linear_interp_finspin(self, finspin):
        # chi is a batch of final spins i.e. torch.Size([n])
        right_spin_idx = torch.bucketize(finspin, self.qnmdata_a)
        right_spin_val = self.qnmdata_a[right_spin_idx]
        # QNMData_a is sorted, hence take the previous index
        left_spin_idx = right_spin_idx - 1
        left_spin_val = self.qnmdata_a[left_spin_idx]

        if not torch.all(left_spin_val < right_spin_val):
            raise RuntimeError(
                "Left value in grid should be greater than right. "
                "Maybe be caused for extremal spin values."
            )
        left_fring = self.qnmdata_fring[left_spin_idx]
        right_fring = self.qnmdata_fring[right_spin_idx]
        slope_fring = right_fring - left_fring
        slope_fring /= right_spin_val - left_spin_val

        left_fdamp = self.qnmdata_fdamp[left_spin_idx]
        right_fdamp = self.qnmdata_fdamp[right_spin_idx]
        slope_fdamp = right_fdamp - left_fdamp
        slope_fdamp /= right_spin_val - left_spin_val

        return (
            slope_fring * (finspin - left_spin_val) + left_fring,
            slope_fdamp * (finspin - left_spin_val) + left_fdamp,
        )

    # Utility functions taken from PhenomD utilities in lalsimulation
    # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomD_internals.c
    def sigma1Fit(self, eta, eta2, xi):
        return (
            2096.551999295543
            + 1463.7493168261553 * eta
            + (
                1312.5493286098522
                + 18307.330017082117 * eta
                - 43534.1440746107 * eta2
                + (
                    -833.2889543511114
                    + 32047.31997183187 * eta
                    - 108609.45037520859 * eta2
                )
                * xi
                + (
                    452.25136398112204
                    + 8353.439546391714 * eta
                    - 44531.3250037322 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def sigma2Fit(self, eta, eta2, xi):
        return (
            -10114.056472621156
            - 44631.01109458185 * eta
            + (
                -6541.308761668722
                - 266959.23419307504 * eta
                + 686328.3229317984 * eta2
                + (
                    3405.6372187679685
                    - 437507.7208209015 * eta
                    + 1.6318171307344697e6 * eta2
                )
                * xi
                + (
                    -7462.648563007646
                    - 114585.25177153319 * eta
                    + 674402.4689098676 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def sigma3Fit(self, eta, eta2, xi):
        return (
            22933.658273436497
            + 230960.00814979506 * eta
            + (
                14961.083974183695
                + 1.1940181342318142e6 * eta
                - 3.1042239693052764e6 * eta2
                + (
                    -3038.166617199259
                    + 1.8720322849093592e6 * eta
                    - 7.309145012085539e6 * eta2
                )
                * xi
                + (
                    42738.22871475411
                    + 467502.018616601 * eta
                    - 3.064853498512499e6 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def sigma4Fit(self, eta, eta2, xi):
        return (
            -14621.71522218357
            - 377812.8579387104 * eta
            + (
                -9608.682631509726
                - 1.7108925257214056e6 * eta
                + 4.332924601416521e6 * eta2
                + (
                    -22366.683262266528
                    - 2.5019716386377467e6 * eta
                    + 1.0274495902259542e7 * eta2
                )
                * xi
                + (
                    -85360.30079034246
                    - 570025.3441737515 * eta
                    + 4.396844346849777e6 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def gamma1_fun(self, eta, eta2, xi):
        return (
            0.006927402739328343
            + 0.03020474290328911 * eta
            + (
                0.006308024337706171
                - 0.12074130661131138 * eta
                + 0.26271598905781324 * eta2
                + (
                    0.0034151773647198794
                    - 0.10779338611188374 * eta
                    + 0.27098966966891747 * eta2
                )
                * xi
                + (
                    0.0007374185938559283
                    - 0.02749621038376281 * eta
                    + 0.0733150789135702 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def gamma2_fun(self, eta, eta2, xi):
        return (
            1.010344404799477
            + 0.0008993122007234548 * eta
            + (
                0.283949116804459
                - 4.049752962958005 * eta
                + 13.207828172665366 * eta2
                + (
                    0.10396278486805426
                    - 7.025059158961947 * eta
                    + 24.784892370130475 * eta2
                )
                * xi
                + (
                    0.03093202475605892
                    - 2.6924023896851663 * eta
                    + 9.609374464684983 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def gamma3_fun(self, eta, eta2, xi):
        return (
            1.3081615607036106
            - 0.005537729694807678 * eta
            + (
                -0.06782917938621007
                - 0.6689834970767117 * eta
                + 3.403147966134083 * eta2
                + (
                    -0.05296577374411866
                    - 0.9923793203111362 * eta
                    + 4.820681208409587 * eta2
                )
                * xi
                + (
                    -0.006134139870393713
                    - 0.38429253308696365 * eta
                    + 1.7561754421985984 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def beta1Fit(self, eta, eta2, xi):
        return (
            97.89747327985583
            - 42.659730877489224 * eta
            + (
                153.48421037904913
                - 1417.0620760768954 * eta
                + 2752.8614143665027 * eta2
                + (
                    138.7406469558649
                    - 1433.6585075135881 * eta
                    + 2857.7418952430758 * eta2
                )
                * xi
                + (
                    41.025109467376126
                    - 423.680737974639 * eta
                    + 850.3594335657173 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def beta2Fit(self, eta, eta2, xi):
        return (
            -3.282701958759534
            - 9.051384468245866 * eta
            + (
                -12.415449742258042
                + 55.4716447709787 * eta
                - 106.05109938966335 * eta2
                + (
                    -11.953044553690658
                    + 76.80704618365418 * eta
                    - 155.33172948098394 * eta2
                )
                * xi
                + (
                    -3.4129261592393263
                    + 25.572377569952536 * eta
                    - 54.408036707740465 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def beta3Fit(self, eta, eta2, xi):
        return (
            -0.000025156429818799565
            + 0.000019750256942201327 * eta
            + (
                -0.000018370671469295915
                + 0.000021886317041311973 * eta
                + 0.00008250240316860033 * eta2
                + (
                    7.157371250566708e-6
                    - 0.000055780000112270685 * eta
                    + 0.00019142082884072178 * eta2
                )
                * xi
                + (
                    5.447166261464217e-6
                    - 0.00003220610095021982 * eta
                    + 0.00007974016714984341 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def alpha1Fit(self, eta, eta2, xi):
        return (
            43.31514709695348
            + 638.6332679188081 * eta
            + (
                -32.85768747216059
                + 2415.8938269370315 * eta
                - 5766.875169379177 * eta2
                + (
                    -61.85459307173841
                    + 2953.967762459948 * eta
                    - 8986.29057591497 * eta2
                )
                * xi
                + (
                    -21.571435779762044
                    + 981.2158224673428 * eta
                    - 3239.5664895930286 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def alpha2Fit(self, eta, eta2, xi):
        return (
            -0.07020209449091723
            - 0.16269798450687084 * eta
            + (
                -0.1872514685185499
                + 1.138313650449945 * eta
                - 2.8334196304430046 * eta2
                + (
                    -0.17137955686840617
                    + 1.7197549338119527 * eta
                    - 4.539717148261272 * eta2
                )
                * xi
                + (
                    -0.049983437357548705
                    + 0.6062072055948309 * eta
                    - 1.682769616644546 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def alpha3Fit(self, eta, eta2, xi):
        return (
            9.5988072383479
            - 397.05438595557433 * eta
            + (
                16.202126189517813
                - 1574.8286986717037 * eta
                + 3600.3410843831093 * eta2
                + (
                    27.092429659075467
                    - 1786.482357315139 * eta
                    + 5152.919378666511 * eta2
                )
                * xi
                + (
                    11.175710130033895
                    - 577.7999423177481 * eta
                    + 1808.730762932043 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def alpha4Fit(self, eta, eta2, xi):
        return (
            -0.02989487384493607
            + 1.4022106448583738 * eta
            + (
                -0.07356049468633846
                + 0.8337006542278661 * eta
                + 0.2240008282397391 * eta2
                + (
                    -0.055202870001177226
                    + 0.5667186343606578 * eta
                    + 0.7186931973380503 * eta2
                )
                * xi
                + (
                    -0.015507437354325743
                    + 0.15750322779277187 * eta
                    + 0.21076815715176228 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def alpha5Fit(self, eta, eta2, xi):
        return (
            0.9974408278363099
            - 0.007884449714907203 * eta
            + (
                -0.059046901195591035
                + 1.3958712396764088 * eta
                - 4.516631601676276 * eta2
                + (
                    -0.05585343136869692
                    + 1.7516580039343603 * eta
                    - 5.990208965347804 * eta2
                )
                * xi
                + (
                    -0.017945336522161195
                    + 0.5965097794825992 * eta
                    - 2.0608879367971804 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def rho1_fun(self, eta, eta2, xi):
        return (
            3931.8979897196696
            - 17395.758706812805 * eta
            + (
                3132.375545898835
                + 343965.86092361377 * eta
                - 1.2162565819981997e6 * eta2
                + (
                    -70698.00600428853
                    + 1.383907177859705e6 * eta
                    - 3.9662761890979446e6 * eta2
                )
                * xi
                + (
                    -60017.52423652596
                    + 803515.1181825735 * eta
                    - 2.091710365941658e6 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def rho2_fun(self, eta, eta2, xi):
        return (
            -40105.47653771657
            + 112253.0169706701 * eta
            + (
                23561.696065836168
                - 3.476180699403351e6 * eta
                + 1.137593670849482e7 * eta2
                + (
                    754313.1127166454
                    - 1.308476044625268e7 * eta
                    + 3.6444584853928134e7 * eta2
                )
                * xi
                + (
                    596226.612472288
                    - 7.4277901143564405e6 * eta
                    + 1.8928977514040343e7 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def rho3_fun(self, eta, eta2, xi):
        return (
            83208.35471266537
            - 191237.7264145924 * eta
            + (
                -210916.2454782992
                + 8.71797508352568e6 * eta
                - 2.6914942420669552e7 * eta2
                + (
                    -1.9889806527362722e6
                    + 3.0888029960154563e7 * eta
                    - 8.390870279256162e7 * eta2
                )
                * xi
                + (
                    -1.4535031953446497e6
                    + 1.7063528990822166e7 * eta
                    - 4.2748659731120914e7 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def FinalSpin0815(self, eta, eta2, chi1, chi2):
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        Seta = torch.nan_to_num(Seta)  # avoid nan around eta = 0.25
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        m1s = m1 * m1
        m2s = m2 * m2
        s = m1s * chi1 + m2s * chi2
        eta3 = eta2 * eta
        s2 = s * s
        s3 = s2 * s
        return eta * (
            3.4641016151377544
            - 4.399247300629289 * eta
            + 9.397292189321194 * eta2
            - 13.180949901606242 * eta3
            + (
                (1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta)
                + (0.1014665242971878 - 2.0967746996832157 * eta) * s
                + (-1.3546806617824356 + 4.108962025369336 * eta) * s2
                + (-0.8676969352555539 + 2.064046835273906 * eta) * s3
            )
            * s
        )

    def PhenomInternal_EradRational0815(self, eta, eta2, chi1, chi2):
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        m1s = m1 * m1
        m2s = m2 * m2
        s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

        eta3 = eta2 * eta

        return (
            eta
            * (
                0.055974469826360077
                + 0.5809510763115132 * eta
                - 0.9606726679372312 * eta2
                + 3.352411249771192 * eta3
            )
            * (
                1.0
                + (
                    -0.0030302335878845507
                    - 2.0066110851351073 * eta
                    + 7.7050567802399215 * eta2
                )
                * s
            )
        ) / (
            1.0
            + (
                -0.6714403054720589
                - 1.4756929437702908 * eta
                + 7.304676214885011 * eta2
            )
            * s
        )

    def AmpIntColFitCoeff(self, eta, eta2, xi):
        return (
            0.8149838730507785
            + 2.5747553517454658 * eta
            + (
                1.1610198035496786
                - 2.3627771785551537 * eta
                + 6.771038707057573 * eta2
                + (
                    0.7570782938606834
                    - 2.7256896890432474 * eta
                    + 7.1140380397149965 * eta2
                )
                * xi
                + (
                    0.1766934149293479
                    - 0.7978690983168183 * eta
                    + 2.1162391502005153 * eta2
                )
                * xi
                * xi
            )
            * xi
        )

    def delta_values(self, f1, f2, f3, v1, v2, v3, d1, d2):
        f12 = f1 * f1
        f13 = f1 * f12
        f14 = f1 * f13
        f15 = f1 * f14
        f22 = f2 * f2
        f23 = f2 * f22
        f24 = f2 * f23
        f32 = f3 * f3
        f33 = f3 * f32
        f34 = f3 * f33
        f35 = f3 * f34
        delta_0 = -(
            (
                d2 * f15 * f22 * f3
                - 2 * d2 * f14 * f23 * f3
                + d2 * f13 * f24 * f3
                - d2 * f15 * f2 * f32
                + d2 * f14 * f22 * f32
                - d1 * f13 * f23 * f32
                + d2 * f13 * f23 * f32
                + d1 * f12 * f24 * f32
                - d2 * f12 * f24 * f32
                + d2 * f14 * f2 * f33
                + 2 * d1 * f13 * f22 * f33
                - 2 * d2 * f13 * f22 * f33
                - d1 * f12 * f23 * f33
                + d2 * f12 * f23 * f33
                - d1 * f1 * f24 * f33
                - d1 * f13 * f2 * f34
                - d1 * f12 * f22 * f34
                + 2 * d1 * f1 * f23 * f34
                + d1 * f12 * f2 * f35
                - d1 * f1 * f22 * f35
                + 4 * f12 * f23 * f32 * v1
                - 3 * f1 * f24 * f32 * v1
                - 8 * f12 * f22 * f33 * v1
                + 4 * f1 * f23 * f33 * v1
                + f24 * f33 * v1
                + 4 * f12 * f2 * f34 * v1
                + f1 * f22 * f34 * v1
                - 2 * f23 * f34 * v1
                - 2 * f1 * f2 * f35 * v1
                + f22 * f35 * v1
                - f15 * f32 * v2
                + 3 * f14 * f33 * v2
                - 3 * f13 * f34 * v2
                + f12 * f35 * v2
                - f15 * f22 * v3
                + 2 * f14 * f23 * v3
                - f13 * f24 * v3
                + 2 * f15 * f2 * f3 * v3
                - f14 * f22 * f3 * v3
                - 4 * f13 * f23 * f3 * v3
                + 3 * f12 * f24 * f3 * v3
                - 4 * f14 * f2 * f32 * v3
                + 8 * f13 * f22 * f32 * v3
                - 4 * f12 * f23 * f32 * v3
            )
            / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (f3 - f2) ** 2)
        )

        delta_1 = -(
            (
                -(d2 * f15 * f22)
                + 2 * d2 * f14 * f23
                - d2 * f13 * f24
                - d2 * f14 * f22 * f3
                + 2 * d1 * f13 * f23 * f3
                + 2 * d2 * f13 * f23 * f3
                - 2 * d1 * f12 * f24 * f3
                - d2 * f12 * f24 * f3
                + d2 * f15 * f32
                - 3 * d1 * f13 * f22 * f32
                - d2 * f13 * f22 * f32
                + 2 * d1 * f12 * f23 * f32
                - 2 * d2 * f12 * f23 * f32
                + d1 * f1 * f24 * f32
                + 2 * d2 * f1 * f24 * f32
                - d2 * f14 * f33
                + d1 * f12 * f22 * f33
                + 3 * d2 * f12 * f22 * f33
                - 2 * d1 * f1 * f23 * f33
                - 2 * d2 * f1 * f23 * f33
                + d1 * f24 * f33
                + d1 * f13 * f34
                + d1 * f1 * f22 * f34
                - 2 * d1 * f23 * f34
                - d1 * f12 * f35
                + d1 * f22 * f35
                - 8 * f12 * f23 * f3 * v1
                + 6 * f1 * f24 * f3 * v1
                + 12 * f12 * f22 * f32 * v1
                - 8 * f1 * f23 * f32 * v1
                - 4 * f12 * f34 * v1
                + 2 * f1 * f35 * v1
                + 2 * f15 * f3 * v2
                - 4 * f14 * f32 * v2
                + 4 * f12 * f34 * v2
                - 2 * f1 * f35 * v2
                - 2 * f15 * f3 * v3
                + 8 * f12 * f23 * f3 * v3
                - 6 * f1 * f24 * f3 * v3
                + 4 * f14 * f32 * v3
                - 12 * f12 * f22 * f32 * v3
                + 8 * f1 * f23 * f32 * v3
            )
            / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (-f2 + f3) ** 2)
        )

        delta_2 = -(
            (
                d2 * f15 * f2
                - d1 * f13 * f23
                - 3 * d2 * f13 * f23
                + d1 * f12 * f24
                + 2 * d2 * f12 * f24
                - d2 * f15 * f3
                + d2 * f14 * f2 * f3
                - d1 * f12 * f23 * f3
                + d2 * f12 * f23 * f3
                + d1 * f1 * f24 * f3
                - d2 * f1 * f24 * f3
                - d2 * f14 * f32
                + 3 * d1 * f13 * f2 * f32
                + d2 * f13 * f2 * f32
                - d1 * f1 * f23 * f32
                + d2 * f1 * f23 * f32
                - 2 * d1 * f24 * f32
                - d2 * f24 * f32
                - 2 * d1 * f13 * f33
                + 2 * d2 * f13 * f33
                - d1 * f12 * f2 * f33
                - 3 * d2 * f12 * f2 * f33
                + 3 * d1 * f23 * f33
                + d2 * f23 * f33
                + d1 * f12 * f34
                - d1 * f1 * f2 * f34
                + d1 * f1 * f35
                - d1 * f2 * f35
                + 4 * f12 * f23 * v1
                - 3 * f1 * f24 * v1
                + 4 * f1 * f23 * f3 * v1
                - 3 * f24 * f3 * v1
                - 12 * f12 * f2 * f32 * v1
                + 4 * f23 * f32 * v1
                + 8 * f12 * f33 * v1
                - f1 * f34 * v1
                - f35 * v1
                - f15 * v2
                - f14 * f3 * v2
                + 8 * f13 * f32 * v2
                - 8 * f12 * f33 * v2
                + f1 * f34 * v2
                + f35 * v2
                + f15 * v3
                - 4 * f12 * f23 * v3
                + 3 * f1 * f24 * v3
                + f14 * f3 * v3
                - 4 * f1 * f23 * f3 * v3
                + 3 * f24 * f3 * v3
                - 8 * f13 * f32 * v3
                + 12 * f12 * f2 * f32 * v3
                - 4 * f23 * f32 * v3
            )
            / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (-f2 + f3) ** 2)
        )

        delta_3 = -(
            (
                -2 * d2 * f14 * f2
                + d1 * f13 * f22
                + 3 * d2 * f13 * f22
                - d1 * f1 * f24
                - d2 * f1 * f24
                + 2 * d2 * f14 * f3
                - 2 * d1 * f13 * f2 * f3
                - 2 * d2 * f13 * f2 * f3
                + d1 * f12 * f22 * f3
                - d2 * f12 * f22 * f3
                + d1 * f24 * f3
                + d2 * f24 * f3
                + d1 * f13 * f32
                - d2 * f13 * f32
                - 2 * d1 * f12 * f2 * f32
                + 2 * d2 * f12 * f2 * f32
                + d1 * f1 * f22 * f32
                - d2 * f1 * f22 * f32
                + d1 * f12 * f33
                - d2 * f12 * f33
                + 2 * d1 * f1 * f2 * f33
                + 2 * d2 * f1 * f2 * f33
                - 3 * d1 * f22 * f33
                - d2 * f22 * f33
                - 2 * d1 * f1 * f34
                + 2 * d1 * f2 * f34
                - 4 * f12 * f22 * v1
                + 2 * f24 * v1
                + 8 * f12 * f2 * f3 * v1
                - 4 * f1 * f22 * f3 * v1
                - 4 * f12 * f32 * v1
                + 8 * f1 * f2 * f32 * v1
                - 4 * f22 * f32 * v1
                - 4 * f1 * f33 * v1
                + 2 * f34 * v1
                + 2 * f14 * v2
                - 4 * f13 * f3 * v2
                + 4 * f1 * f33 * v2
                - 2 * f34 * v2
                - 2 * f14 * v3
                + 4 * f12 * f22 * v3
                - 2 * f24 * v3
                + 4 * f13 * f3 * v3
                - 8 * f12 * f2 * f3 * v3
                + 4 * f1 * f22 * f3 * v3
                + 4 * f12 * f32 * v3
                - 8 * f1 * f2 * f32 * v3
                + 4 * f22 * f32 * v3
            )
            / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (-f2 + f3) ** 2)
        )

        delta_4 = -(
            (
                d2 * f13 * f2
                - d1 * f12 * f22
                - 2 * d2 * f12 * f22
                + d1 * f1 * f23
                + d2 * f1 * f23
                - d2 * f13 * f3
                + 2 * d1 * f12 * f2 * f3
                + d2 * f12 * f2 * f3
                - d1 * f1 * f22 * f3
                + d2 * f1 * f22 * f3
                - d1 * f23 * f3
                - d2 * f23 * f3
                - d1 * f12 * f32
                + d2 * f12 * f32
                - d1 * f1 * f2 * f32
                - 2 * d2 * f1 * f2 * f32
                + 2 * d1 * f22 * f32
                + d2 * f22 * f32
                + d1 * f1 * f33
                - d1 * f2 * f33
                + 3 * f1 * f22 * v1
                - 2 * f23 * v1
                - 6 * f1 * f2 * f3 * v1
                + 3 * f22 * f3 * v1
                + 3 * f1 * f32 * v1
                - f33 * v1
                - f13 * v2
                + 3 * f12 * f3 * v2
                - 3 * f1 * f32 * v2
                + f33 * v2
                + f13 * v3
                - 3 * f1 * f22 * v3
                + 2 * f23 * v3
                - 3 * f12 * f3 * v3
                + 6 * f1 * f2 * f3 * v3
                - 3 * f22 * f3 * v3
            )
            / ((f1 - f2) ** 2 * (f1 - f3) ** 3 * (-f2 + f3) ** 2)
        )

        return delta_0, delta_1, delta_2, delta_3, delta_4

    def chiPN(self, Seta, eta, chi1, chi2):
        chi_s = chi1 + chi2
        chi_a = chi1 - chi2

        return 0.5 * (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
