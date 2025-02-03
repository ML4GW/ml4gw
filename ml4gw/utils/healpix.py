from typing import Tuple

import numpy as np
import torch

from ..constants import PI
from ..types import HealpixIndex


def nest2uniq(
    nside: HealpixIndex,
    ipix: HealpixIndex,
) -> HealpixIndex:
    return 4 * nside * nside + ipix


def nside2npix(nside: HealpixIndex) -> HealpixIndex:
    return 12 * nside * nside


def nside2pixarea(nside: HealpixIndex, degrees: bool = False) -> HealpixIndex:
    pixarea = 4 * PI / nside2npix(nside)

    if degrees:
        pixarea = pixarea * (180.0 / PI) ** 2

    return pixarea


def lonlat2thetaphi(
    lon: HealpixIndex, lat: HealpixIndex
) -> Tuple[HealpixIndex, HealpixIndex]:
    if isinstance(lon, torch.Tensor) and isinstance(lat, torch.Tensor):
        return PI / 2.0 - torch.deg2rad(lat), torch.deg2rad(lon)
    else:
        return PI / 2.0 - np.deg2rad(lat), np.deg2rad(lon)
