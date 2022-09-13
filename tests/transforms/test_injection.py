from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from ml4gw.transforms import RandomWaveformInjection


@pytest.fixture(params=[128, 512])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[["H1"], ["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.mark.parametrize("factor", [None, 1, 2, 0.5])
def test_fit(sample_rate, ifos, factor):
    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.df = 1 / 8

    background = {i: np.random.randn(2048) for i in ifos}
    fit_sample_rate = sample_rate * factor if factor is not None else None
    RandomWaveformInjection.fit(mock, fit_sample_rate, **background)

    assert mock.tensors.shape == (len(ifos), 3, 3)
    assert mock.vertices.shape == (len(ifos), 3)
    assert mock.background.shape == (len(ifos), 4 * sample_rate + 1)

    # use this background as our target for passing things
    # other than numpy arrays
    target_background = mock.background.cpu().numpy()

    # now try passing in TimeSeries objects with differing sample rates
    ts_sample_rate = fit_sample_rate or sample_rate
    background = {
        i: TimeSeries(x, dt=1 / ts_sample_rate) for i, x in background.items()
    }

    # be extra safe by creating a new mock
    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.df = 1 / 8
    RandomWaveformInjection.fit(mock, **background)
    assert np.isclose(mock.background, target_background, rtol=1e-6).all()

    # now try passing in pre-computed psds
    background = {
        i: x.resample(sample_rate).psd(2, method="median", window="hann")
        for i, x in background.items()
    }

    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.df = 1 / 8
    RandomWaveformInjection.fit(mock, **background)
    assert np.isclose(mock.background, target_background, rtol=1e-6).all()


def test_sample_source_param():
    param = torch.arange(10, 20)
    idx = torch.arange(2, 5)
    result = RandomWaveformInjection._sample_source_param(
        None, param, idx, None
    )
    result = result.cpu().numpy()
    assert (result == np.arange(12, 15)).all()

    mock = MagicMock()
    mock.tensors.device = "cpu"
    result = RandomWaveformInjection._sample_source_param(
        mock, lambda N: torch.arange(5, 5 + N), None, 4
    )
    result = result.cpu().numpy()
    assert (result == np.arange(5, 9)).all()


@pytest.fixture(autouse=True)
def compute_observed_strain(ifos):
    def f(*args, **kwargs):
        waveform = kwargs["plus"] + kwargs["cross"]
        return torch.stack([waveform + i for i in range(len(ifos))], axis=1)

    with patch("ml4gw.gw.compute_observed_strain", new=f):
        yield


def reweight_snrs(*args, **kwargs):
    return args[0]


@patch("ml4gw.gw.reweight_snrs", new=reweight_snrs)
def test_sample(sample_rate, ifos):
    mock = MagicMock()

    # first make sure we enforce fitting
    mock.background = None
    with pytest.raises(TypeError) as exc:
        RandomWaveformInjection.sample(mock, 1)
    assert str(exc.value).endswith("WaveformSampler.fit first")
    mock.background = MagicMock()

    # now give our dummy sampler some waveforms to sample
    mock.num_waveforms = 100
    mock._sample_source_param = lambda p, i, N: (
        RandomWaveformInjection._sample_source_param(mock, p, i, N)
    )
    mock.tensors.device = "cpu"

    waveforms = {i: torch.randn(100, 1024) for i in ["plus", "cross"]}
    mock.polarizations = waveforms

    # since our patched functions just add the polarizations,
    # we'll create an array of expected outputs up front
    waveform = waveforms["plus"] + waveforms["cross"]
    summed = torch.stack([waveform + i for i in range(len(ifos))], axis=1)

    # first test with passing indices
    idx = torch.arange(5, 50, 5)
    result, params = RandomWaveformInjection.sample(mock, idx)
    assert (result == summed[idx]).all()
    mock.dec.assert_called_with(len(idx))

    # now test with calling -1 to make sure
    # we get everything back in order
    result, params = RandomWaveformInjection.sample(mock, -1)
    assert len(result) == len(summed)
    assert (result == summed).all()
    mock.dec.assert_called_with(100)

    # make sure if we ask for too many waveforms we get yelled at
    with pytest.raises(ValueError):
        RandomWaveformInjection.sample(mock, 101)

    # now supply a regular allowed integer and patch
    # torch.randperm so we know what we're getting
    idx = torch.arange(10, 40, 5)
    with patch("torch.randperm", return_value=idx):
        result, params = RandomWaveformInjection.sample(mock, len(idx))
    assert (result == summed[idx]).all()
    mock.dec.assert_called_with(len(idx))


@pytest.fixture(params=[0.5, 1])
def prob(request):
    return request.param


@patch("ml4gw.gw.reweight_snrs", new=reweight_snrs)
def test_random_waveform_injection(prob, ifos):
    waveforms = {i: torch.randn(100, 1024) for i in ["plus", "cross"]}
    waveform = waveforms["plus"] + waveforms["cross"]
    summed = torch.stack([waveform + i for i in range(len(ifos))], axis=1)

    dec = MagicMock()
    psi = MagicMock()
    phi = MagicMock()
    snr = MagicMock()

    # enforce all polarizations have to be same length
    with pytest.raises(ValueError) as exc:
        wrong = {str(i): np.random.randn(i + 2, 1024) for i in range(2)}
        transform = RandomWaveformInjection(
            dec, psi, phi, snr, sample_rate=1024, prob=prob, **wrong
        )
    assert str(exc.value).startswith("Polarization")

    # enforce prob is greater than 0
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            dec, psi, phi, snr, sample_rate=1024, prob=0, **waveforms
        )
    assert str(exc.value).startswith("Injection probability")

    # same for leq 1
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            dec, psi, phi, snr, sample_rate=1024, prob=1.2, **waveforms
        )
    assert str(exc.value).startswith("Injection probability")

    # now make it for real
    transform = RandomWaveformInjection(
        dec, psi, phi, snr, sample_rate=1024, prob=prob, **waveforms
    )
    assert len(list(transform.parameters())) == 2
    assert transform.num_waveforms == 100
    assert transform.df == 1
    assert transform.mask is None

    # make background not None so we can sample
    transform.background = MagicMock()
    transform.tensors = MagicMock()
    transform.tensors.device = "cpu"

    # create some fake data to inject into
    X = torch.zeros((16, len(ifos), 256))
    y = torch.zeros((16,))

    # now patch a few torch random functions so
    # we know what outputs to expect
    expected_count = 16 if prob == 1 else 9
    mask = torch.arange(0, 0.9, 0.9 / 16)
    rand_patch = patch("torch.rand", return_value=mask)
    perm_patch = patch("torch.randperm", return_value=torch.arange(16))
    randint_patch = patch(
        "torch.randint", return_value=torch.arange(expected_count)
    )

    # run the transform's forward method with the patchees
    with rand_patch as rand_mock, perm_patch as perm_mock:
        with randint_patch as randint_mock:
            X_hat, y_hat = transform(X, y)

    # ensure all the expected calls happened
    rand_mock.assert_called_with(size=(16,))
    perm_mock.assert_called_with(100)
    randint_mock.assert_called_with(256, 512, size=(expected_count,))
    dec.assert_called_with(expected_count)

    # now verify that the data matches up: the first
    # `exepcted_count` rows should be injected, and
    # the rest should be all 0s
    for i, (x_row, y_row) in enumerate(zip(X_hat, y_hat)):
        if i >= expected_count:
            assert (x_row == 0).all()
            assert (y_row == 0).all()
        else:
            assert (x_row == summed[i, :, i : i + 256]).all()
            assert (y_row == 1).all()

    # now try things with one of the parameters as a tensor
    dec = np.random.randn(100)

    # first make sure that having a param
    # too short raises an exception
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            dec[:99], psi, phi, snr, sample_rate=1024, prob=prob, **waveforms
        )
    assert str(exc.value).startswith("Source parameter 'dec' is not")

    # now verify that the tensor is added as a true parameter
    transform = RandomWaveformInjection(
        dec, psi, phi, snr, sample_rate=1024, prob=prob, **waveforms
    )
    assert len(list(transform.parameters())) == 3
    transform.tensors = MagicMock()
    transform.tensors.device = "cpu"

    transform.background = MagicMock()
    with rand_patch as rand_mock, perm_patch as perm_mock:
        with randint_patch as randint_mock:
            X_hat, y_hat = transform(X, y)

    # TODO: how to make sure this works?
    # expected_arg = torch.Tensor(dec[:expected_count])
    # compute_observed_strain.assert_called_with(expected_arg)

    for i, (x_row, y_row) in enumerate(zip(X_hat, y_hat)):
        if i >= expected_count:
            assert (x_row == 0).all()
            assert (y_row == 0).all()
        else:
            assert (x_row == 2 * summed[i, :, i : i + 256]).all()
            assert (y_row == 1).all()
