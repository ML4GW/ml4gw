from typing import Union

import torch

from ..constants import PI


def nest2uniq(
    nside: Union[int, torch.Tensor], ipix: Union[int, torch.Tensor]
) -> Union[int, torch.Tensor]:
    return 4 * nside * nside + ipix


def nside2npix(nside: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
    return 12 * nside * nside


def nside2pixarea(
    nside: Union[int, torch.Tensor], degrees: bool = True
) -> Union[float, torch.Tensor]:
    pixarea = 4 * PI / nside2npix(nside)

    if degrees:
        pixarea = pixarea * (180.0 / PI) ** 2

    return pixarea


def lonlat2thetaphi(
    lon: Union[int, torch.Tensor], lat: Union[int, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    lon, lat = torch.asarray(lon), torch.asarray(lat)
    return PI / 2.0 - torch.deg2rad(lat), torch.deg2rad(lon)
