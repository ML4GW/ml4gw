import torch

from ..constants import MAX_NSIDE, PI


def nest2uniq(nside, ipix):
    return 4 * nside * nside + ipix


def nside2npix(nside):
    return 12 * nside * nside


def nside2pixarea(nside, degrees=True):
    pixarea = 4 * PI / nside2npix(nside)

    if degrees:
        pixarea = pixarea * (180.0 / PI) ** 2

    return pixarea


def isnsideok(nside, nest=False):
    if hasattr(nside, "__len__"):
        if not isinstance(nside, torch.Tensor):
            nside = torch.asarray(nside)
        is_nside_ok = (
            (nside == nside.int()) & (nside > 0) & (nside <= MAX_NSIDE)
        )
        if nest:
            is_nside_ok &= nside.int() & (nside.int() - 1 == 0)
    else:
        is_nside_ok = (nside == int(nside)) and (0 < nside <= MAX_NSIDE)
        if nest:
            is_nside_ok = is_nside_ok and (int(nside) & (int(nside) - 1)) == 0
    return is_nside_ok


def check_nside(nside, nest=False):
    """Raises exception if nside is not valid"""
    if not torch.all(isnsideok(nside, nest=nest)):
        raise ValueError(
            f"{nside} is not a valid nside parameter (must be a power of 2,\
                less than 2**30)"
        )


def lonlat2thetaphi(lon, lat):
    return PI / 2.0 - torch.deg2rad(lat), torch.deg2rad(lon)


def check_theta_valid(theta):
    """Raises exception if theta is not within 0 and pi"""
    theta = torch.asarray(theta)
    if not ((theta >= 0).all() and (theta <= PI + 1e-5).all()):
        raise ValueError("THETA is out of range [0,pi]")
