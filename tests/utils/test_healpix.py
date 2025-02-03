import healpy as hp
import pytest
import torch
from torch.distributions import Uniform

import ml4gw.utils.healpix as mlhp


@pytest.fixture(params=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def nside(request):
    return request.param


@pytest.fixture()
def long(request):
    dist = Uniform(-180, 180)
    return dist.sample((10,))


@pytest.fixture()
def lat(request):
    dist = Uniform(-90, 90)
    return dist.sample((10,))


def test_nside2npix(nside):
    assert mlhp.nside2npix(nside) == hp.nside2npix(nside)


def test_nside2pixarea(nside):
    assert mlhp.nside2pixarea(nside) == hp.nside2pixarea(nside)


def test_lonlat2thetaphi(long, lat):
    theta_ml4gw, phi_ml4gw = mlhp.lonlat2thetaphi(long, lat)
    theta, phi = hp.pixelfunc.lonlat2thetaphi(long, lat)
    assert torch.allclose(theta_ml4gw, theta)
    assert torch.allclose(phi_ml4gw, phi)


def test_nest2uniq(nside):
    ipix = torch.randint(0, mlhp.nside2npix(nside), (10,))
    assert torch.allclose(
        mlhp.nest2uniq(nside, ipix), hp.nest2uniq(nside, ipix)
    )
