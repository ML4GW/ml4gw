import torch
from jaxtyping import Float

from ...constants import MTSUN_SI
from ...types import BatchTensor, FrequencySeries1d
from .phenom_d import IMRPhenomD


class IMRPhenomDECO(IMRPhenomD):
    def forward(
        self,
        f: FrequencySeries1d,
        chirp_mass: BatchTensor,
        mass_ratio: BatchTensor,
        chi1: BatchTensor,
        chi2: BatchTensor,
        compactness: BatchTensor,
        distance: BatchTensor,
        phic: BatchTensor,
        inclination: BatchTensor,
        f_ref: float,
        **kwargs,
    ):
        """
        IMRPhenomDECO waveform

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
            compactness:
                effective compactness of binary at contact
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
            f,
            chirp_mass,
            mass_ratio,
            chi1,
            chi2,
            compactness,
            distance,
            phic,
            f_ref,
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
        compactness: BatchTensor,
        distance: BatchTensor,
        phic: BatchTensor,
        f_ref: float,
    ) -> Float[FrequencySeries1d, " batch"]:
        # PhenomDECO reuses the PhenomD phase model and the amplitude model
        # is parametrized by an effective compactness of the two objects;
        # see Phys. Rev. D 112, 104017 (2025) for details

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

        # compactness fixed at 0.5, as phase unchanged wrt PhenomD
        Mf_peak_phase = self.fmaxCalc_deco(fRD, fDM, gamma2, gamma3, 0.5)
        _, t0 = self.phenom_d_mrd_phase(
            Mf_peak_phase, eta, eta2, chi1, chi2, xi
        )

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

        amp, _ = self.phenom_deco_amp(
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
            compactness,
        )

        amp_0 = self.taylorf2_amplitude(
            Mf, mass_1, mass_2, eta, distance
        )  # this includes f^(-7/6) dependence

        h0 = -amp_0 * amp * torch.exp(-1j * Psi)
        return h0

    def phenom_deco_amp(
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
        compactness,
        fRD=None,  # used for passing ringdown frequency from phenom_p
        fDM=None,  # used for passing damping frequency from phenom_p
    ):
        ins_amp, ins_Damp = self.phenom_d_inspiral_amp(
            Mf, eta, eta2, Seta, xi, chi1, chi2, chi12, chi22
        )
        int_amp, int_Damp = self.phenom_deco_int_amp(
            Mf,
            eta,
            eta2,
            Seta,
            chi1,
            chi2,
            chi12,
            chi22,
            xi,
            compactness,
            fRD,
            fDM,
        )
        mrd_amp, mrd_Damp = self.phenom_deco_mrd_amp(
            Mf, eta, eta2, chi1, chi2, xi, compactness, fRD, fDM
        )

        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)

        # merger ringdown
        if (fRD is None) != (fDM is None):
            raise ValueError(
                "Both fRD and fDM must either be provided or both be None"
            )
        if (fRD is None) and (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)

        Mf_peak = self.fmaxCalc_deco(fRD, fDM, gamma2, gamma3, compactness)

        # Geometric peak and joining frequencies
        Mf_peak = (torch.ones_like(Mf).mT * Mf_peak).mT
        Mf_join_ins = (
            0.014
            * (torch.ones_like(Mf).mT * (2 * compactness) ** (3 / 2.0)).mT
        )
        # construct full IMR Amp
        theta_minus_f1 = torch.heaviside(
            Mf_join_ins - Mf, torch.tensor(0.0, device=Mf.device)
        )
        theta_plus_f1 = torch.heaviside(
            Mf - Mf_join_ins, torch.tensor(1.0, device=Mf.device)
        )
        theta_minus_f2 = torch.heaviside(
            Mf_peak - Mf, torch.tensor(1.0, device=Mf.device)
        )
        theta_plus_f2 = torch.heaviside(
            Mf - Mf_peak, torch.tensor(0.0, device=Mf.device)
        )

        amp = theta_minus_f1 * ins_amp
        amp += theta_plus_f1 * int_amp * theta_minus_f2
        amp += theta_plus_f2 * mrd_amp

        Damp = theta_minus_f1 * ins_Damp
        Damp += theta_plus_f1 * int_Damp * theta_minus_f2
        Damp += theta_plus_f2 * mrd_Damp
        return amp, Damp

    def phenom_deco_int_amp(
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
        compactness,
        fRD=None,  # used for passing ringdown frequency from phenom_p
        fDM=None,  # used for passing damping frequency from phenom_p
    ):
        # merger ringdown
        if (fRD is None) != (fDM is None):
            raise ValueError(
                "Both fRD and fDM must either be provided or both be None"
            )
        if (fRD is None) and (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)

        # Geometric frequency definition from PhenomD header file
        AMP_fJoin_INS = 0.014 * (2 * compactness) ** (3 / 2.0)

        Mf1 = (AMP_fJoin_INS * torch.ones_like(Mf).mT).mT
        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)

        fpeak = self.fmaxCalc_deco(fRD, fDM, gamma2, gamma3, compactness)
        Mf3 = (torch.ones_like(Mf).mT * fpeak).mT
        dfx = 0.5 * (Mf3 - Mf1)
        Mf2 = Mf1 + dfx

        v1, d1 = self.phenom_d_inspiral_amp(
            Mf1, eta, eta2, Seta, xi, chi1, chi2, chi12, chi22
        )
        v3, d2 = self.phenom_deco_mrd_amp(
            Mf3,
            eta,
            eta2,
            chi1,
            chi2,
            xi,
            compactness,
            fRD,
            fDM,
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

    def phenom_deco_mrd_amp(
        self,
        Mf,
        eta,
        eta2,
        chi1,
        chi2,
        xi,
        compactness,
        fRD=None,
        fDM=None,
    ):
        # merger ringdown
        if (fRD is None) != (fDM is None):
            raise ValueError(
                "Both fRD and fDM must either be provided or both be None"
            )
        if (fRD is None) and (fDM is None):
            fRD, fDM = self.fring_fdamp(eta, eta2, chi1, chi2)

        fRD_deco = fRD * (2 * compactness) ** (3 / 2.0)

        gamma1 = self.gamma1_fun(eta, eta2, xi)
        gamma2 = self.gamma2_fun(eta, eta2, xi)
        gamma3 = self.gamma3_fun(eta, eta2, xi)
        fDMgamma3 = fDM * gamma3
        pow2_fDMgamma3 = (torch.ones_like(Mf).mT * fDMgamma3 * fDMgamma3).mT
        fminfRD = Mf - (torch.ones_like(Mf).mT * fRD_deco).mT
        exp_times_lorentzian = torch.exp(fminfRD.mT * gamma2 / fDMgamma3).mT
        exp_times_lorentzian *= fminfRD**2 + pow2_fDMgamma3

        amp = (1 / exp_times_lorentzian.mT * gamma1 * gamma3 * fDM).mT
        Damp = (fminfRD.mT * -2 * fDM * gamma1 * gamma3) / (
            fminfRD * fminfRD + pow2_fDMgamma3
        ).mT - (gamma2 * gamma1)
        Damp = Damp.mT / exp_times_lorentzian
        return amp, Damp

    def fmaxCalc_deco(self, fRD, fDM, gamma2, gamma3, compactness):
        mask = gamma2 <= 1
        # calculate result for gamma2 <= 1 case
        sqrt_term = torch.sqrt(1 - gamma2.pow(2))
        fRD_deco = fRD * (2 * compactness) ** (3 / 2.0)
        result_case1 = fRD_deco + (fDM * (-1 + sqrt_term) * gamma3) / gamma2

        # calculate result for gamma2 > 1 case
        # i.e. don't add sqrt term
        result_case2 = fRD_deco + (-fDM * gamma3) / gamma2

        # combine results using mask
        result = torch.where(mask, result_case1, result_case2)

        return torch.abs(result)
