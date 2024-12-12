import torch

from ml4gw.constants import MTSUN_SI, PI
from ml4gw.types import BatchTensor


def rotate_z(angle: BatchTensor, x, y, z):
    x_tmp = x * torch.cos(angle) - y * torch.sin(angle)
    y_tmp = x * torch.sin(angle) + y * torch.cos(angle)
    return x_tmp, y_tmp, z


def rotate_y(angle, x, y, z):
    x_tmp = x * torch.cos(angle) + z * torch.sin(angle)
    z_tmp = -x * torch.sin(angle) + z * torch.cos(angle)
    return x_tmp, y, z_tmp


def XLALSimInspiralLN(
    total_mass: BatchTensor, eta: BatchTensor, v: BatchTensor
):
    return total_mass**2 * eta / v


def XLALSimInspiralL_2PN(eta: BatchTensor):
    return 1.5 + eta / 6.0


def bilby_spins_to_lalsim(
    thetajn: BatchTensor,
    phijl: BatchTensor,
    theta1: BatchTensor,
    theta2: BatchTensor,
    phi12: BatchTensor,
    chi1: BatchTensor,
    chi2: BatchTensor,
    mass_1: BatchTensor,
    mass_2: BatchTensor,
    f_ref: float,
    phi_ref: BatchTensor,
):
    """
    See

    Args:
        thetaJN:
            zenith angle between J and N (rad)
        phiJL:
           azimuthal angle of L_N on its cone about J
        theta1:
        theta2:
        phi12:
        chi1:
        chi2:
        mass_1:
    """

    # check if f_ref is valid
    if f_ref == 0.0:
        raise ValueError(
            "f_ref = 0 is invalid. "
            "Please pass in the starting GW frequency instead."
        )

    # starting frame: LNhat is along the z-axis and the unit
    # spin vectors are defined from the angles relative to LNhat.
    # Note that we put s1hat in the x-z plane, and phi12
    # sets the azimuthal angle of s2hat measured from the x-axis.
    lnh_x = 0
    lnh_y = 0
    lnh_z = 1
    # Spins are given wrt to L,
    # but still we cannot fill the spin as we do not know
    # what will be the relative orientation of L and N.
    # Note that these spin components are NOT wrt to binary
    # separation vector, but wrt to binary separation vector
    # at phiref=0.

    s1hatx = torch.sin(theta1) * torch.cos(phi_ref)
    s1haty = torch.sin(theta1) * torch.sin(phi_ref)
    s1hatz = torch.cos(theta1)
    s2hatx = torch.sin(theta2) * torch.cos(phi12 + phi_ref)
    s2haty = torch.sin(theta2) * torch.sin(phi12 + phi_ref)
    s2hatz = torch.cos(theta2)

    total_mass = mass_1 + mass_2

    eta = mass_1 * mass_2 / (mass_1 + mass_2) / (mass_1 + mass_2)

    # v parameter at reference point
    v0 = ((mass_1 + mass_2) * MTSUN_SI * PI * f_ref) ** (1 / 3)

    # Define S1, S2, J with proper magnitudes */

    l_mag = XLALSimInspiralLN(total_mass, eta, v0) * (
        1.0 + v0 * v0 * XLALSimInspiralL_2PN(eta)
    )
    s1x = mass_1 * mass_1 * chi1 * s1hatx
    s1y = mass_1 * mass_1 * chi1 * s1haty
    s1z = mass_1 * mass_1 * chi1 * s1hatz
    s2x = mass_2 * mass_2 * chi2 * s2hatx
    s2y = mass_2 * mass_2 * chi2 * s2haty
    s2z = mass_2 * mass_2 * chi2 * s2hatz
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = l_mag + s1z + s2z

    # Normalize J to Jhat, find its angles in starting frame */
    Jnorm = torch.sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    theta0 = torch.acos(Jhatz)
    phi0 = torch.atan2(Jhaty, Jhatx)

    # Rotation 1: Rotate about z-axis by -phi0 to put Jhat in x-z plane
    s1hatx, s1haty, s1hatz = rotate_z(-phi0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phi0, s2hatx, s2haty, s2hatz)

    # Rotation 2: Rotate about new y-axis by -theta0
    # to put Jhat along z-axis

    lnh_x, lnh_y, lnh_z = rotate_y(-theta0, lnh_x, lnh_y, lnh_z)
    s1hatx, s1haty, s1hatz = rotate_y(-theta0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-theta0, s2hatx, s2haty, s2hatz)

    # Rotation 3: Rotate about new z-axis by phiJL to put L at desired
    # azimuth about J. Note that is currently in x-z plane towards -x
    # (i.e. azimuth=pi). Hence we rotate about z by phiJL - LAL_PI
    lnh_x, lnh_y, lnh_z = rotate_z(phijl - PI, lnh_x, lnh_y, lnh_z)
    s1hatx, s1haty, s1hatz = rotate_z(phijl - PI, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(phijl - PI, s2hatx, s2haty, s2hatz)

    # The cosinus of the angle between L and N is the scalar
    # product of the two vectors.
    # We do not need to perform additional rotation to compute it.
    Nx = 0.0
    Ny = torch.sin(thetajn)
    Nz = torch.cos(thetajn)
    incl = torch.acos(Nx * lnh_x + Ny * lnh_y + Nz * lnh_z)

    # Rotation 4-5: Now J is along z and N in y-z plane, inclined from J
    # by thetaJN and with >ve component along y.
    # Now we bring L into the z axis to get spin components.
    thetalj = torch.acos(lnh_z)
    phil = torch.atan2(lnh_y, lnh_x)

    s1hatx, s1haty, s1hatz = rotate_z(-phil, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_z(-phil, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_z(-phil, Nx, Ny, Nz)

    s1hatx, s1haty, s1hatz = rotate_y(-thetalj, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = rotate_y(-thetalj, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz = rotate_y(-thetalj, Nx, Ny, Nz)

    # Rotation 6: Now L is along z and we have to bring N
    # in the y-z plane with >ve y components.

    phiN = torch.atan2(Ny, Nx)
    # Note the extra -phiRef here:
    # output spins must be given wrt to two body separations
    # which are rigidly rotated with spins
    s1hatx, s1haty, s1hatz = rotate_z(
        PI / 2.0 - phiN - phi_ref, s1hatx, s1haty, s1hatz
    )
    s2hatx, s2haty, s2hatz = rotate_z(
        PI / 2.0 - phiN - phi_ref, s2hatx, s2haty, s2hatz
    )

    s1x = s1hatx * chi1
    s1y = s1haty * chi1
    s1z = s1hatz * chi1
    s2x = s2hatx * chi2
    s2y = s2haty * chi2
    s2z = s2hatz * chi2

    return incl, s1x, s1y, s1z, s2x, s2y, s2z
