from collections import OrderedDict

import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from ml4gw.transforms import Whitening


@pytest.fixture(params=[512, 1024])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[0.5, 1])
def fduration(request):
    return request.param


@pytest.fixture(params=[2, 4])
def fftlength(request):
    return request.param


@pytest.fixture(params=[0, 20])
def highpass(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def test_whitening_transform(
    num_ifos, sample_rate, fduration, fftlength, highpass, dtype
):
    tform = Whitening(num_ifos, sample_rate, fduration, dtype=dtype)

    # make sure that trying to run the forward
    # call without fitting raises an error
    kernel_length = fduration * 2
    kernel_size = int(kernel_length * sample_rate)
    X = torch.randn(8, num_ifos, kernel_size).type(dtype)
    X = 1e-19 + 1e-20 * X
    with pytest.raises(ValueError) as exc_info:
        tform(X)
    assert str(exc_info.value).startswith("Must fit parameters")

    # fit to some random background
    ifos = "hlv"[:num_ifos]
    backgrounds = OrderedDict(
        [(i, np.random.randn(100 * sample_rate)) for i in ifos]
    )
    backgrounds = {k: 1e-19 + 1e-20 * v for k, v in backgrounds.items()}

    # verify that if our kernel length isn't greater
    # than the length of fduration we raise an error
    with pytest.raises(ValueError) as exc_info:
        tform.fit(fduration, fftlength, highpass, **backgrounds)
    assert str(exc_info.value).startswith("Whitening pad size")

    # now fit to background and actually do the whitening
    tform.fit(kernel_length, fftlength, highpass, **backgrounds)
    results = tform(X).cpu().numpy()

    # check to make sure that whitened data is 0 mean unit variance
    # TODO: there are better statistically-motivated values to use
    # here, but we have to make the bounds so loose they ultimately
    # lose most meaning
    mean = results.mean()
    std = results.std()
    assert np.isclose(mean, 0, atol=0.02)
    assert np.isclose(std, 1, atol=0.1)

    # pre-compute the background asds for comparison
    asds = {}
    for ifo, x in backgrounds.items():
        x = TimeSeries(x, sample_rate=sample_rate)
        x = x.asd(fftlength, method="median", window="hann")
        asds[ifo] = x

    # for each ifo in each batch element, verify that whitening
    # with gwpy produces roughly the same result
    pad = int(sample_rate * fduration / 2)
    for x, y in zip(X.cpu().numpy(), results):
        for input, result, ifo in zip(x, y, ifos):
            asd = asds[ifo]
            ts = TimeSeries(input, sample_rate=sample_rate)
            ts = ts.whiten(
                asd=asd, fduration=fduration, highpass=highpass, window="hann"
            )
            ts = ts.value[pad:-pad]

            # TODO: this isn't the strongest check in the world,
            # but it's the closest I can get for now. Check that
            # at least 95% of the outputs are within 1% of the
            # value produced by gwpy
            close = np.isclose(result, ts, rtol=1e-2)
            assert close.mean() > 0.95


def test_whitening_save_and_load(dtype, tmp_path):
    tform = Whitening(2, 512, 1, dtype)
    h1 = torch.randn(512 * 10)
    l1 = torch.randn(512 * 10)
    tform.fit(2, 2, h1=h1, l1=l1)

    assert (tform.kernel_length == 2).all().item()
    assert tform.built
    value = tform.time_domain_filter

    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(tform.state_dict(), tmp_path / "weights.pt")

    tform = Whitening(2, 512, 1, dtype)
    assert not tform.built
    assert (tform.kernel_length == 0).all().item()

    tform.load_state_dict(torch.load(tmp_path / "weights.pt"))
    assert tform.built
    assert (tform.kernel_length == 2).all().item()
    assert (tform.time_domain_filter == value).all().item()

    # ensure that loading something with the wrong shape
    # causes an error to get raised
    tform = Whitening(2, 512, 0.5, dtype)
    with pytest.raises(RuntimeError):
        tform.load_state_dict(torch.load(tmp_path / "weights.pt"))
