import numpy as np
import pytest
import torch
from gwpy.signal.qtransform import QPlane
from gwpy.signal.qtransform import QTile as gwpy_QTile
from gwpy.signal.qtransform import QTiling

from ml4gw.transforms import QScan, SingleQTransform
from ml4gw.transforms.qtransform import QTile


@pytest.fixture(params=[1, 2])
def duration(request):
    return request.param


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=["mean", "median", None])
def norm(request):
    return request.param


@pytest.fixture(params=[64, 128])
def num_f_bins(request):
    return request.param


@pytest.fixture(params=[64, 128])
def num_t_bins(request):
    return request.param


@pytest.fixture(params=[12, 100])
def q(request):
    return request.param


@pytest.fixture(params=[0.2, 0.35])
def mismatch(request):
    return request.param


@pytest.fixture(params=[128, 512])
def frequency(request):
    return request.param


def test_qtile(
    q,
    frequency,
    duration,
    sample_rate,
    mismatch,
    norm,
):

    X = torch.randn(int(duration * sample_rate))
    X = torch.fft.rfft(X, norm="forward")
    X[..., 1:] *= 2

    torch_qtile = QTile(q, frequency, duration, sample_rate, mismatch)
    gwpy_qtile = gwpy_QTile(q, frequency, duration, sample_rate, mismatch)

    assert torch_qtile.ntiles() == gwpy_qtile.ntiles
    assert np.allclose(
        torch_qtile.get_window().numpy(), gwpy_qtile.get_window()
    )
    assert (
        torch_qtile.get_data_indices().numpy() == gwpy_qtile.get_data_indices()
    ).all()
    assert np.allclose(
        torch_qtile(X, norm).numpy(),
        gwpy_qtile.transform(X, norm, 0),
        rtol=1e-3,
    )


def test_singleqtransform(
    q,
    duration,
    sample_rate,
    mismatch,
    norm,
    num_f_bins,
    num_t_bins,
):
    X = torch.randn(int(duration * sample_rate))
    fseries = torch.fft.rfft(X, norm="forward")
    fseries[..., 1:] *= 2

    qtransform = SingleQTransform(
        duration, sample_rate, q, frange=[0, torch.inf], mismatch=mismatch
    )

    with pytest.raises(RuntimeError):
        qtransform.get_max_energy()

    with pytest.raises(RuntimeError):
        qtransform.interpolate(num_f_bins, num_t_bins)

    qplane = QPlane(
        q,
        frange=[0, np.inf],
        duration=duration,
        sampling=sample_rate,
        mismatch=mismatch,
    )

    assert (
        qtransform.get_freqs().numpy() == list(qplane._iter_frequencies())
    ).all()

    qtransform.compute_qtiles(X, norm)
    torch_qtiles = qtransform.qtiles
    gwpy_qtiles = [qtile.transform(fseries, norm, 0) for qtile in qplane]

    for t, g in zip(torch_qtiles, gwpy_qtiles):
        assert np.allclose(t.numpy(), g, rtol=1e-3)

    transformed = qtransform(X, num_f_bins, num_t_bins, norm)
    assert transformed.shape[-2] == num_f_bins
    assert transformed.shape[-1] == num_t_bins


def test_get_qs(
    duration,
    sample_rate,
    mismatch,
):
    frange = [0, torch.inf]
    qrange = [1, 1000]

    qscan = QScan(duration, sample_rate, qrange, frange, mismatch)
    qtiling = QTiling(
        duration, sample_rate, qrange, frange=[0, np.inf], mismatch=mismatch
    )

    assert np.allclose(qscan.get_qs(), qtiling.qs)
