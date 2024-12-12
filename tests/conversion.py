import numpy as np
import torch
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from torch.distributions import Uniform

from ml4gw.constants import MSUN
from ml4gw.waveforms.conversion import bilby_spins_to_lalsim


def test_bilby_to_lalsim_spins():
    thetajn = Uniform(0, torch.pi).sample((100,))
    phijl = Uniform(0, torch.pi).sample((100,))
    theta1 = Uniform(0, torch.pi).sample((100,))
    theta2 = Uniform(0, torch.pi).sample((100,))
    phi12 = Uniform(0, torch.pi).sample((100,))
    chi1 = Uniform(0, 0.99).sample((100,))
    chi2 = Uniform(0, 0.99).sample((100,))
    mass_1 = Uniform(5, 100).sample((100,))
    mass_2 = Uniform(5, 100).sample((100,))
    f_ref = 40.0
    phi_ref = Uniform(0, torch.pi).sample((100,))
    incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        thetajn,
        phijl,
        theta1,
        theta2,
        phi12,
        chi1,
        chi2,
        mass_1,
        mass_2,
        f_ref,
        phi_ref,
    )
    for i in range(2):

        (
            lal_incl,
            lal_s1x,
            lal_s1y,
            lal_s1z,
            lal_s2x,
            lal_s2y,
            lal_s2z,
        ) = SimInspiralTransformPrecessingNewInitialConditions(
            thetajn[i].item(),
            phijl[i].item(),
            theta1[i].item(),
            theta2[i].item(),
            phi12[i].item(),
            chi1[i].item(),
            chi2[i].item(),
            mass_1[i].item() * MSUN,
            mass_2[i].item() * MSUN,
            f_ref,
            phi_ref[i].item(),
        )

        # check if the values are close up to 4 decimal places
        assert np.isclose(incl[i].item(), lal_incl, atol=1e-4)
        assert np.isclose(s1x[i].item(), lal_s1x, atol=1e-4)
        assert np.isclose(s1y[i].item(), lal_s1y, atol=1e-4)
        assert np.isclose(s1z[i].item(), lal_s1z, atol=1e-4)
        assert np.isclose(s2x[i].item(), lal_s2x, atol=1e-4)
        assert np.isclose(s2y[i].item(), lal_s2y, atol=1e-4)
        assert np.isclose(s2z[i].item(), lal_s2z, atol=1e-4)
