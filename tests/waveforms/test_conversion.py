import numpy as np
import torch
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from torch.distributions import Uniform

from ml4gw.constants import MSUN
from ml4gw.waveforms.conversion import bilby_spins_to_lalsim


def test_bilby_to_lalsim_spins():
    theta_jn = Uniform(0, torch.pi).sample((100,))
    phi_jl = Uniform(0, 2 * torch.pi).sample((100,))
    tilt_1 = Uniform(0, torch.pi).sample((100,))
    tilt_2 = Uniform(0, torch.pi).sample((100,))
    phi_12 = Uniform(0, 2 * torch.pi).sample((100,))
    a_1 = Uniform(0, 0.99).sample((100,))
    a_2 = Uniform(0, 0.99).sample((100,))
    mass_1 = Uniform(3, 100).sample((100,))
    mass_2 = Uniform(3, 100).sample((100,))
    f_ref = 40.0
    phi_ref = Uniform(0, torch.pi).sample((100,))
    incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
        theta_jn,
        phi_jl,
        tilt_1,
        tilt_2,
        phi_12,
        a_1,
        a_2,
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
            theta_jn[i].item(),
            phi_jl[i].item(),
            tilt_1[i].item(),
            tilt_2[i].item(),
            phi_12[i].item(),
            a_1[i].item(),
            a_2[i].item(),
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
