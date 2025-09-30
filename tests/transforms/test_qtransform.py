import numpy as np
import pytest
import torch
from gwpy.signal.qtransform import QPlane, QTiling
from gwpy.signal.qtransform import QTile as gwpy_QTile

from ml4gw.transforms import QScan, SingleQTransform
from ml4gw.transforms.qtransform import QTile


@pytest.fixture(params=[1, 2])
def duration(request):
    return request.param


@pytest.fixture(params=[1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=["mean", "median", None])
def norm(request):
    return request.param


@pytest.fixture(params=[[64, 64], [64, 128], [128, 128]])
def spectrogram_shape(request):
    return request.param


@pytest.fixture(params=[12, 50])
def q(request):
    return request.param


@pytest.fixture(params=[0.2, 0.35])
def mismatch(request):
    return request.param


@pytest.fixture(params=[128, 256])
def frequency(request):
    return request.param


@pytest.fixture(params=["bilinear", "bicubic", "spline"])
def interpolation_method(request):
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

    X = torch.randn(2, 2, 2, int(sample_rate * duration))
    with pytest.raises(ValueError):
        torch_qtile(X)


def test_singleqtransform(
    q,
    duration,
    sample_rate,
    mismatch,
    norm,
    spectrogram_shape,
    interpolation_method,
):
    X = torch.randn(int(duration * sample_rate))
    fseries = torch.fft.rfft(X, norm="forward")
    fseries[..., 1:] *= 2

    with pytest.raises(ValueError):
        qtransform = SingleQTransform(
            duration,
            sample_rate,
            spectrogram_shape,
            q,
            frange=[0, torch.inf],
            mismatch=mismatch,
            interpolation_method="nonsense",
        )

    with pytest.raises(ValueError):
        qtransform = SingleQTransform(
            duration,
            sample_rate,
            spectrogram_shape,
            q=1000,
            frange=[0, torch.inf],
            mismatch=mismatch,
            interpolation_method="nonsense",
        )

    qtransform = SingleQTransform(
        duration,
        sample_rate,
        spectrogram_shape,
        q,
        frange=[0, torch.inf],
        mismatch=mismatch,
        interpolation_method=interpolation_method,
    )

    with pytest.raises(RuntimeError):
        qtransform.get_max_energy()

    with pytest.raises(RuntimeError):
        qtransform.interpolate()

    qplane = QPlane(
        q,
        frange=[0, np.inf],
        duration=duration,
        sampling=sample_rate,
        mismatch=mismatch,
    )

    assert (qtransform.freqs.numpy() == qplane.frequencies).all()

    qtransform.compute_qtiles(X, norm)
    torch_qtiles = qtransform.qtiles
    gwpy_qtiles = [qtile.transform(fseries, norm, 0) for qtile in qplane]

    for t, g in zip(torch_qtiles, gwpy_qtiles, strict=True):
        assert np.allclose(t.numpy(), g, rtol=1e-3)

    transformed = qtransform(X, norm)
    assert list(transformed.shape[-2:]) == spectrogram_shape


def test_get_qs(
    duration,
    sample_rate,
    mismatch,
    spectrogram_shape,
):
    frange = [0, torch.inf]
    qrange = [1, 1000]

    qscan = QScan(
        duration,
        sample_rate,
        spectrogram_shape,
        qrange,
        frange,
        mismatch=mismatch,
    )
    qtiling = QTiling(
        duration, sample_rate, qrange, frange=[0, np.inf], mismatch=mismatch
    )

    assert np.allclose(qscan.get_qs(), qtiling.qs)

    # Just check that the QScan runs
    data = torch.randn(int(sample_rate * duration))
    _ = qscan(data)
