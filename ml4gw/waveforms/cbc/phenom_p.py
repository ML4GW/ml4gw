"""
    Based on the JAX implementation of IMRPhenomPv2 from
    https://github.com/tedwards2412/ripple/blob/main/src/ripplegw/waveforms/IMRPhenomPv2.py
"""

from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from ...constants import MPC_SEC, MTSUN_SI, PI
from ...types import BatchTensor, FrequencySeries1d
from ..conversion import rotate_y, rotate_z
from .phenom_d import IMRPhenomD


class IMRPhenomPv2(IMRPhenomD):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        fs: FrequencySeries1d,
        chirp_mass: BatchTensor,
        mass_ratio: BatchTensor,
        s1x: BatchTensor,
        s1y: BatchTensor,
        s1z: BatchTensor,
        s2x: BatchTensor,
        s2y: BatchTensor,
        s2z: BatchTensor,
        distance: BatchTensor,
        phic: BatchTensor,
        inclination: BatchTensor,
        f_ref: float,
        tc: Optional[BatchTensor] = None,
    ):
        """
        IMRPhenomPv2 waveform

        Args:
            fs :
                Frequency series in Hz.
            chirp_mass :
                Chirp mass in solar masses.
            mass_ratio :
                Mass ratio m1/m2.
            s1x :
                Spin component x of the first BH.
            s1y :
                Spin component y of the first BH.
            s1z :
                Spin component z of the first BH.
            s2x :
                Spin component x of the second BH.
            s2y :
                Spin component y of the second BH.
            s2z :
                Spin component z of the second BH.
            distance :
                Luminosity distance in Mpc.
            tc :
                Coalescence time.
            phic :
                Reference phase.
            inclination :
                Inclination angle.
            f_ref :
                Reference frequency in Hz.

        Returns:
            hc, hp: Tuple[torch.Tensor, torch.Tensor]
                Cross and plus polarizations

        Note: m1 must be larger than m2.
        """

        if tc is None:
            tc = torch.zeros_like(chirp_mass)

        m2 = chirp_mass * (1.0 + mass_ratio) ** 0.2 / mass_ratio**0.6
        m1 = m2 * mass_ratio

        # # flip m1 m2. For some reason LAL uses this convention for PhenomPv2
        m1, m2 = m2, m1
        s1x, s2x = s2x, s1x
        s1y, s2y = s2y, s1y
        s1z, s2z = s2z, s1z
        (
            chi1_l,
            chi2_l,
            chip,
            thetaJN,
            alpha0,
            phi_aligned,
            zeta_polariz,
        ) = self.convert_spins(
            m1, m2, f_ref, phic, inclination, s1x, s1y, s1z, s2x, s2y, s2z
        )

        phic = 2 * phi_aligned
        q = m2 / m1  # q>=1
        M = m1 + m2
        chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
        chil = (1.0 + q) / q * chi_eff
        eta = m1 * m2 / (M * M)
        eta2 = eta * eta
        Seta = torch.sqrt(1.0 - 4.0 * eta)
        chi = self.chiPN(Seta, eta, chi2_l, chi1_l)
        chi22 = chi2_l * chi2_l
        chi12 = chi1_l * chi1_l
        xi = -1.0 + chi
        m_sec = M * MTSUN_SI
        piM = PI * m_sec

        omega_ref = piM * f_ref
        logomega_ref = torch.log(omega_ref)
        omega_ref_cbrt = (piM * f_ref) ** (1 / 3)  # == v0
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        angcoeffs = self.ComputeNNLOanglecoeffs(q, chil, chip)

        alphaNNLOoffset = (
            angcoeffs["alphacoeff1"] / omega_ref
            + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
            + angcoeffs["alphacoeff3"] / omega_ref_cbrt
            + angcoeffs["alphacoeff4"] * logomega_ref
            + angcoeffs["alphacoeff5"] * omega_ref_cbrt
        )

        epsilonNNLOoffset = (
            angcoeffs["epsiloncoeff1"] / omega_ref
            + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
            + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
            + angcoeffs["epsiloncoeff4"] * logomega_ref
            + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt
        )

        Y2m2 = self.SpinWeightedY(thetaJN, 0, -2, 2, -2)
        Y2m1 = self.SpinWeightedY(thetaJN, 0, -2, 2, -1)
        Y20 = self.SpinWeightedY(thetaJN, 0, -2, 2, -0)
        Y21 = self.SpinWeightedY(thetaJN, 0, -2, 2, 1)
        Y22 = self.SpinWeightedY(thetaJN, 0, -2, 2, 2)
        Y2 = torch.stack((Y2m2, Y2m1, Y20, Y21, Y22))

        hPhenomDs, diffRDphase = self.PhenomPOneFrequency(
            fs,
            m2,
            m1,
            eta,
            eta2,
            Seta,
            chi2_l,
            chi1_l,
            chi12,
            chi22,
            chip,
            phic,
            M,
            xi,
            distance,
        )

        hp, hc = self.PhenomPCoreTwistUp(
            fs,
            hPhenomDs,
            eta,
            chi1_l,
            chi2_l,
            chip,
            M,
            angcoeffs,
            Y2,
            alphaNNLOoffset - alpha0,
            epsilonNNLOoffset,
        )
        t0 = (diffRDphase.unsqueeze(1)) / (2 * PI)
        phase_corr = torch.cos(2 * PI * fs * (t0)) - 1j * torch.sin(
            2 * PI * fs * (t0)
        )
        M_s = (m1 + m2) * MTSUN_SI
        phase_corr_tc = torch.exp(
            -1j * fs * M_s.unsqueeze(1) * tc.unsqueeze(1)
        )
        hp *= phase_corr * phase_corr_tc
        hc *= phase_corr * phase_corr_tc

        c2z = torch.cos(2 * zeta_polariz).unsqueeze(1)
        s2z = torch.sin(2 * zeta_polariz).unsqueeze(1)
        hplus = c2z * hp + s2z * hc
        hcross = c2z * hc - s2z * hp
        return hcross, hplus

    def PhenomPCoreTwistUp(
        self,
        fHz: FrequencySeries1d,
        hPhenom: BatchTensor,
        eta: BatchTensor,
        chi1_l: BatchTensor,
        chi2_l: BatchTensor,
        chip: BatchTensor,
        M: BatchTensor,
        angcoeffs: Dict[str, BatchTensor],
        Y2m: BatchTensor,
        alphaoffset: BatchTensor,
        epsilonoffset: BatchTensor,
    ) -> Tuple[BatchTensor, BatchTensor]:
        assert angcoeffs is not None
        assert Y2m is not None
        f = fHz * MTSUN_SI * M.unsqueeze(1)  # Frequency in geometric units
        q = (1.0 + torch.sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
        m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
        m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
        Sperp = chip * (
            m2 * m2
        )  # Dimensionfull spin component in the orbital plane.
        # S_perp = S_2_perp chi_eff = m1 * chi1_l + m2 * chi2_l
        # effective spin for M=1

        SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

        omega = PI * f
        logomega = torch.log(omega)
        omega_cbrt = (omega) ** (1 / 3)
        omega_cbrt2 = omega_cbrt * omega_cbrt
        alpha = (
            (
                angcoeffs["alphacoeff1"] / omega.mT
                + angcoeffs["alphacoeff2"] / omega_cbrt2.mT
                + angcoeffs["alphacoeff3"] / omega_cbrt.mT
                + angcoeffs["alphacoeff4"] * logomega.mT
                + angcoeffs["alphacoeff5"] * omega_cbrt.mT
            )
            - alphaoffset
        ).mT

        epsilon = (
            (
                angcoeffs["epsiloncoeff1"] / omega.mT
                + angcoeffs["epsiloncoeff2"] / omega_cbrt2.mT
                + angcoeffs["epsiloncoeff3"] / omega_cbrt.mT
                + angcoeffs["epsiloncoeff4"] * logomega.mT
                + angcoeffs["epsiloncoeff5"] * omega_cbrt.mT
            )
            - epsilonoffset
        ).mT

        cBetah, sBetah = self.WignerdCoefficients(
            omega_cbrt.mT, SL, eta, Sperp
        )

        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah2 * cBetah
        cBetah4 = cBetah3 * cBetah
        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah2 * sBetah
        sBetah4 = sBetah3 * sBetah

        hp_sum = 0
        hc_sum = 0

        cexp_i_alpha = torch.exp(1j * alpha)
        cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
        cexp_mi_alpha = 1.0 / cexp_i_alpha
        cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
        T2m = (
            cexp_2i_alpha.mT * cBetah4.mT * Y2m[0]
            - cexp_i_alpha.mT * 2 * cBetah3.mT * sBetah.mT * Y2m[1]
            + 1
            * torch.sqrt(torch.tensor(6))
            * sBetah2.mT
            * cBetah2.mT
            * Y2m[2]
            - cexp_mi_alpha.mT * 2 * cBetah.mT * sBetah3.mT * Y2m[3]
            + cexp_m2i_alpha.mT * sBetah4.mT * Y2m[4]
        ).mT
        Tm2m = (
            cexp_m2i_alpha.mT * sBetah4.mT * torch.conj(Y2m[0])
            + cexp_mi_alpha.mT
            * 2
            * cBetah.mT
            * sBetah3.mT
            * torch.conj(Y2m[1])
            + 1
            * torch.sqrt(torch.tensor(6))
            * sBetah2.mT
            * cBetah2.mT
            * torch.conj(Y2m[2])
            + cexp_i_alpha.mT * 2 * cBetah3.mT * sBetah.mT * torch.conj(Y2m[3])
            + cexp_2i_alpha.mT * cBetah4.mT * torch.conj(Y2m[4])
        ).mT
        hp_sum = T2m + Tm2m
        hc_sum = 1j * (T2m - Tm2m)

        eps_phase_hP = torch.exp(-2j * epsilon) * hPhenom / 2.0

        hp = eps_phase_hP * hp_sum
        hc = eps_phase_hP * hc_sum

        return hp, hc

    def PhenomPOneFrequency(
        self,
        fs,
        m1,
        m2,
        eta,
        eta2,
        Seta,
        chi1,
        chi2,
        chi12,
        chi22,
        chip,
        phic,
        M,
        xi,
        distance,
    ):
        """
        m1, m2: in solar masses
        phic: Orbital phase at peak of the underlying non precessing model
        M: Total mass (Solar masses)
        """
        M_s = M * MTSUN_SI
        Mf = torch.outer(M_s, fs)
        fRD, fDM = self.phP_get_fRD_fdamp(m1, m2, chi1, chi2, chip)
        # pass M_s * ringdown and M_s * damping frequency to PhenomD functions
        MfRD, MfDM = M_s * fRD, M_s * fDM

        phase, _ = self.phenom_d_phase(
            Mf, m1, m2, eta, eta2, chi1, chi2, xi, MfRD, MfDM
        )
        phase = (phase.mT - (phic + PI / 4.0)).mT
        # why are they subtracting 2*phic?
        # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomP.c#L1316

        Amp = self.phenom_d_amp(
            Mf,
            m1,
            m2,
            eta,
            eta2,
            Seta,
            chi1,
            chi2,
            chi12,
            chi22,
            xi,
            distance,
            MfRD,
            MfDM,
        )[0]
        Amp0 = self.get_Amp0(Mf, eta)
        dist_s = distance * MPC_SEC
        Amp = ((Amp0 * Amp).mT * (M_s**2.0) / dist_s).mT

        hPhenom = Amp * (torch.exp(-1j * phase))

        # calculating derivative of phase with frequency following
        # https://git.ligo.org/lscsoft/lalsuite/-/blame/master/lalsimulation/lib/LALSimIMRPhenomP.c?page=2#L1057 # noqa: E501
        n_fixed = 1000
        x = torch.linspace(0.8, 1.2, n_fixed, device=fRD.device)
        fRDs = torch.outer(fRD, x)
        delta_fRds = (1.2 * fRD - 0.8 * fRD) / (n_fixed - 1)
        MfRDs = torch.zeros_like(fRDs)
        for i in range(fRD.shape[0]):
            MfRDs[i, :] = torch.outer(M_s, fRDs[i, :])[i, :]
        RD_phase = self.phenom_d_phase(
            MfRDs, m1, m2, eta, eta2, chi1, chi2, xi, MfRD, MfDM
        )[0]
        diff = torch.diff(RD_phase, axis=1)
        diffRDphase = (diff[:, 1:] + diff[:, :-1]) / (
            2 * delta_fRds.unsqueeze(1)
        )
        # interpolate at x = 1, as thats the same as f = fRD
        diffRDphase = -self.interpolate(
            torch.tensor([1]), x[1:-1], diffRDphase
        )
        return hPhenom, diffRDphase

    # Utility functions

    def interpolate(
        self,
        x: Float[Tensor, " new_series"],
        xp: Float[Tensor, " series"],
        fp: Float[Tensor, " series"],
    ) -> Float[Tensor, " new_series"]:
        """One-dimensional linear interpolation for monotonically
        increasing sample points.

        Returns the one-dimensional piecewise linear interpolant to a function
        with given data points :math:`(xp, fp)`, evaluated at :math:`x`

        Args:
            x: the :math:`x`-coordinates at which to evaluate the interpolated
                values.
            xp: the :math:`x`-coordinates of data points, must be increasing.
            fp: the :math:`y`-coordinates of data points, same length as `xp`.

        Returns:
            the interpolated values, same size as `x`.
        """
        original_shape = x.shape
        x = x.flatten()
        xp = xp.flatten()
        fp = fp.flatten()

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])  # slope
        b = fp[:-1] - (m * xp[:-1])

        indices = torch.searchsorted(xp, x, right=False) - 1

        interpolated = m[indices] * x + b[indices]

        return interpolated.reshape(original_shape)

    def L2PNR(
        self,
        v: BatchTensor,
        eta: BatchTensor,
    ) -> BatchTensor:
        eta2 = eta**2
        x = v**2
        x2 = x**2
        tmp = (
            eta
            * (
                1.0
                + (1.5 + eta / 6.0) * x
                + (3.375 - (19.0 * eta) / 8.0 - eta2 / 24.0) * x2
            )
        ) / x**0.5

        return tmp

    def convert_spins(
        self,
        m1: BatchTensor,
        m2: BatchTensor,
        f_ref: float,
        phic: BatchTensor,
        inclination: BatchTensor,
        s1x: BatchTensor,
        s1y: BatchTensor,
        s1z: BatchTensor,
        s2x: BatchTensor,
        s2y: BatchTensor,
        s2z: BatchTensor,
    ) -> Tuple[
        BatchTensor,
        BatchTensor,
        BatchTensor,
        BatchTensor,
        BatchTensor,
        BatchTensor,
        BatchTensor,
    ]:
        M = m1 + m2
        m1_2 = m1 * m1
        m2_2 = m2 * m2
        eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

        # From the components in the source frame, we can easily determine
        # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
        # We also compute the spherical angles of J,
        # which we need to transform to the J frame

        # Aligned spins
        chi1_l = s1z  # Dimensionless aligned spin on BH 1
        chi2_l = s2z  # Dimensionless aligned spin on BH 2

        # Magnitude of the spin projections in the orbital plane
        S1_perp = m1_2 * torch.sqrt(s1x**2 + s1y**2)
        S2_perp = m2_2 * torch.sqrt(s2x**2 + s2y**2)

        A1 = 2 + (3 * m2) / (2 * m1)
        A2 = 2 + (3 * m1) / (2 * m2)
        ASp1 = A1 * S1_perp
        ASp2 = A2 * S2_perp
        num = torch.maximum(ASp1, ASp2)
        den = A2 * m2_2  # warning: this assumes m2 > m1
        chip = num / den

        m_sec = M * MTSUN_SI
        piM = PI * m_sec
        v_ref = (piM * f_ref) ** (1 / 3)
        L0 = M * M * self.L2PNR(v_ref, eta)
        J0x_sf = m1_2 * s1x + m2_2 * s2x
        J0y_sf = m1_2 * s1y + m2_2 * s2y
        J0z_sf = L0 + m1_2 * s1z + m2_2 * s2z
        J0 = torch.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

        thetaJ_sf = torch.arccos(J0z_sf / J0)

        phiJ_sf = torch.arctan2(J0y_sf, J0x_sf)

        phi_aligned = -phiJ_sf

        # First we determine kappa
        # in the source frame, the components of N are given in
        # Eq (35c) of T1500606-v6
        Nx_sf = torch.sin(inclination) * torch.cos(PI / 2.0 - phic)
        Ny_sf = torch.sin(inclination) * torch.sin(PI / 2.0 - phic)
        Nz_sf = torch.cos(inclination)

        tmp_x = Nx_sf
        tmp_y = Ny_sf
        tmp_z = Nz_sf

        tmp_x, tmp_y, tmp_z = rotate_z(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_y(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

        kappa = -torch.arctan2(tmp_y, tmp_x)

        # Then we determine alpha0, by rotating LN
        tmp_x, tmp_y, tmp_z = 0, 0, 1
        tmp_x, tmp_y, tmp_z = rotate_z(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_y(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_z(kappa, tmp_x, tmp_y, tmp_z)

        alpha0 = torch.arctan2(tmp_y, tmp_x)

        # Finally we determine thetaJ, by rotating N
        tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
        tmp_x, tmp_y, tmp_z = rotate_z(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_y(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_z(kappa, tmp_x, tmp_y, tmp_z)
        Nx_Jf, Nz_Jf = tmp_x, tmp_z
        thetaJN = torch.arccos(Nz_Jf)

        # Finally, we need to redefine the polarizations:
        # PhenomP's polarizations are defined following Arun et al
        # (arXiv:0810.5336)
        # i.e. projecting the metric onto the P,Q,N triad defined with
        # P=NxJ/|NxJ| (see (2.6) in there).
        # By contrast, the triad X,Y,N used in LAL
        # ("waveframe" in the nomenclature of T1500606-v6)
        # is defined in e.g. eq (35) of this document
        # (via its components in the source frame;
        # note we use the default Omega=Pi/2).
        # Both triads differ from each other by a rotation around N by an angle
        # \zeta and we need to rotate the polarizations accordingly by 2\zeta

        Xx_sf = -torch.cos(inclination) * torch.sin(phic)
        Xy_sf = -torch.cos(inclination) * torch.cos(phic)
        Xz_sf = torch.sin(inclination)
        tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
        tmp_x, tmp_y, tmp_z = rotate_z(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_y(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = rotate_z(kappa, tmp_x, tmp_y, tmp_z)

        # Now the tmp_a are the components of X in the J frame
        # We need the polar angle of that vector in the P,Q basis of Arun et al
        # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J
        # frame
        PArunx_Jf = 0.0
        PAruny_Jf = -1.0
        PArunz_Jf = 0.0

        # Q = NxP
        QArunx_Jf = Nz_Jf
        QAruny_Jf = 0.0
        QArunz_Jf = -Nx_Jf

        # Calculate the dot products XdotPArun and XdotQArun
        XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
        XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

        zeta_polariz = torch.arctan2(XdotQArun, XdotPArun)
        return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz

    # TODO: add input and output types
    def SpinWeightedY(self, theta, phi, s, l, m):  # noqa: E741
        "copied from SphericalHarmonics.c in LAL"
        if s == -2:
            if l == 2:  # noqa: E741
                if m == -2:
                    fac = (
                        torch.sqrt(torch.tensor(5.0 / (64.0 * PI)))
                        * (1.0 - torch.cos(theta))
                        * (1.0 - torch.cos(theta))
                    )
                elif m == -1:
                    fac = (
                        torch.sqrt(torch.tensor(5.0 / (16.0 * PI)))
                        * torch.sin(theta)
                        * (1.0 - torch.cos(theta))
                    )
                elif m == 0:
                    fac = (
                        torch.sqrt(torch.tensor(15.0 / (32.0 * PI)))
                        * torch.sin(theta)
                        * torch.sin(theta)
                    )
                elif m == 1:
                    fac = (
                        torch.sqrt(torch.tensor(5.0 / (16.0 * PI)))
                        * torch.sin(theta)
                        * (1.0 + torch.cos(theta))
                    )
                elif m == 2:
                    fac = (
                        torch.sqrt(torch.tensor(5.0 / (64.0 * PI)))
                        * (1.0 + torch.cos(theta))
                        * (1.0 + torch.cos(theta))
                    )
                else:
                    raise ValueError(
                        f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l"
                    )
                return fac * torch.complex(
                    torch.cos(torch.tensor(m * phi)),
                    torch.sin(torch.tensor(m * phi)),
                )

    def WignerdCoefficients(
        self,
        v: BatchTensor,
        SL: BatchTensor,
        eta: BatchTensor,
        Sp: BatchTensor,
    ) -> Tuple[BatchTensor, BatchTensor]:
        # We define the shorthand s := Sp / (L + SL)
        L = self.L2PNR(v, eta)
        s = (Sp / (L + SL)).mT
        s2 = s**2
        cos_beta = 1.0 / (1.0 + s2) ** 0.5
        cos_beta_half = ((1.0 + cos_beta) / 2.0) ** 0.5  # cos(beta/2)
        sin_beta_half = ((1.0 - cos_beta) / 2.0) ** 0.5  # sin(beta/2)

        return cos_beta_half, sin_beta_half

    def ComputeNNLOanglecoeffs(
        self,
        q: BatchTensor,
        chil: BatchTensor,
        chip: BatchTensor,
    ) -> Dict[str, BatchTensor]:
        m2 = q / (1.0 + q)
        m1 = 1.0 / (1.0 + q)
        dm = m1 - m2
        mtot = 1.0
        eta = m1 * m2  # mtot = 1
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        mtot2 = mtot * mtot
        mtot4 = mtot2 * mtot2
        mtot6 = mtot4 * mtot2
        mtot8 = mtot6 * mtot2
        chil2 = chil * chil
        chip2 = chip * chip
        chip4 = chip2 * chip2
        dm2 = dm * dm
        dm3 = dm2 * dm
        m2_2 = m2 * m2
        m2_3 = m2_2 * m2
        m2_4 = m2_3 * m2
        m2_5 = m2_4 * m2
        m2_6 = m2_5 * m2
        m2_7 = m2_6 * m2
        m2_8 = m2_7 * m2

        angcoeffs = {}
        angcoeffs["alphacoeff1"] = -0.18229166666666666 - (5 * dm) / (
            64.0 * m2
        )

        angcoeffs["alphacoeff2"] = (-15 * dm * m2 * chil) / (
            128.0 * mtot2 * eta
        ) - (35 * m2_2 * chil) / (128.0 * mtot2 * eta)

        angcoeffs["alphacoeff3"] = (
            -1.7952473958333333
            - (4555 * dm) / (7168.0 * m2)
            - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
            - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
            - (515 * eta) / 384.0
            - (15 * dm2 * eta) / (256.0 * m2_2)
            - (175 * dm * eta) / (256.0 * m2)
        )

        angcoeffs["alphacoeff4"] = (
            -(35 * PI) / 48.0
            - (5 * dm * PI) / (16.0 * m2)
            + (5 * dm2 * chil) / (16.0 * mtot2)
            + (5 * dm * m2 * chil) / (3.0 * mtot2)
            + (2545 * m2_2 * chil) / (1152.0 * mtot2)
            - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
            - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
            + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
            + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
        )

        angcoeffs["alphacoeff5"] = (
            4.318908476114694
            + (27895885 * dm) / (2.1676032e7 * m2)
            - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
            - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
            - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
            + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
            + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
            + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
            + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
            + (39695 * eta) / 86016.0
            + (1615 * dm2 * eta) / (28672.0 * m2_2)
            - (265 * dm * eta) / (14336.0 * m2)
            + (955 * eta2) / 576.0
            + (15 * dm3 * eta2) / (1024.0 * m2_3)
            + (35 * dm2 * eta2) / (256.0 * m2_2)
            + (2725 * dm * eta2) / (3072.0 * m2)
            - (15 * dm * m2 * PI * chil) / (16.0 * mtot2 * eta)
            - (35 * m2_2 * PI * chil) / (16.0 * mtot2 * eta)
            + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
            + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
            + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
            + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
            + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
        )

        angcoeffs["epsiloncoeff1"] = -0.18229166666666666 - (5 * dm) / (
            64.0 * m2
        )
        angcoeffs["epsiloncoeff2"] = (-15 * dm * m2 * chil) / (
            128.0 * mtot2 * eta
        ) - (35 * m2_2 * chil) / (128.0 * mtot2 * eta)
        angcoeffs["epsiloncoeff3"] = (
            -1.7952473958333333
            - (4555 * dm) / (7168.0 * m2)
            - (515 * eta) / 384.0
            - (15 * dm2 * eta) / (256.0 * m2_2)
            - (175 * dm * eta) / (256.0 * m2)
        )
        angcoeffs["epsiloncoeff4"] = (
            -(35 * PI) / 48.0
            - (5 * dm * PI) / (16.0 * m2)
            + (5 * dm2 * chil) / (16.0 * mtot2)
            + (5 * dm * m2 * chil) / (3.0 * mtot2)
            + (2545 * m2_2 * chil) / (1152.0 * mtot2)
            + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
            + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
        )
        angcoeffs["epsiloncoeff5"] = (
            4.318908476114694
            + (27895885 * dm) / (2.1676032e7 * m2)
            + (39695 * eta) / 86016.0
            + (1615 * dm2 * eta) / (28672.0 * m2_2)
            - (265 * dm * eta) / (14336.0 * m2)
            + (955 * eta2) / 576.0
            + (15 * dm3 * eta2) / (1024.0 * m2_3)
            + (35 * dm2 * eta2) / (256.0 * m2_2)
            + (2725 * dm * eta2) / (3072.0 * m2)
            - (15 * dm * m2 * PI * chil) / (16.0 * mtot2 * eta)
            - (35 * m2_2 * PI * chil) / (16.0 * mtot2 * eta)
            + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
            + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
            + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
        )
        return angcoeffs

    def FinalSpin_inplane(
        self,
        m1: BatchTensor,
        m2: BatchTensor,
        chi1_l: BatchTensor,
        chi2_l: BatchTensor,
        chip: BatchTensor,
    ) -> BatchTensor:
        M = m1 + m2
        eta = m1 * m2 / (M * M)
        eta2 = eta * eta
        # m1 > m2, the convention used in phenomD
        # (not the convention of internal phenomP)
        q_factor = m1 / M
        af_parallel = self.FinalSpin0815(eta, eta2, chi1_l, chi2_l)
        Sperp = chip * q_factor * q_factor
        af = torch.copysign(
            torch.ones_like(af_parallel), af_parallel
        ) * torch.sqrt(Sperp * Sperp + af_parallel * af_parallel)
        return af

    def phP_get_fRD_fdamp(
        self, m1, m2, chi1_l, chi2_l, chip
    ) -> Tuple[BatchTensor, BatchTensor]:
        # m1 > m2 should hold here
        finspin = self.FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip)
        m1_s = m1 * MTSUN_SI
        m2_s = m2 * MTSUN_SI
        M_s = m1_s + m2_s
        eta_s = m1_s * m2_s / (M_s**2.0)
        eta_s2 = eta_s * eta_s
        Erad = self.PhenomInternal_EradRational0815(
            eta_s, eta_s2, chi1_l, chi2_l
        )
        fRD = self.interpolate(finspin, self.qnmdata_a, self.qnmdata_fring) / (
            1.0 - Erad
        )
        fdamp = self.interpolate(
            finspin, self.qnmdata_a, self.qnmdata_fdamp
        ) / (1.0 - Erad)
        return fRD / M_s, fdamp / M_s

    def get_Amp0(self, fM_s: BatchTensor, eta: BatchTensor) -> BatchTensor:
        Amp0 = (
            (2.0 / 3.0 * eta.unsqueeze(1)) ** (1.0 / 2.0)
            * (fM_s) ** (-7.0 / 6.0)
            * PI ** (-1.0 / 6.0)
        )
        return Amp0
