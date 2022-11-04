from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

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
    background = {i: np.random.randn(2048) for i in ifos}
    fit_sample_rate = sample_rate * factor if factor is not None else None

    # first test that TypeError gets raised
    # if .fit is called with snr as None
    mock.snr = None
    with pytest.raises(TypeError):
        RandomWaveformInjection.fit(mock, fit_sample_rate, **background)

    # reset mock
    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.df = 1 / 8

    RandomWaveformInjection.fit(mock, fit_sample_rate, **background)
    assert mock.background.shape == (len(ifos), 4 * sample_rate + 1)
    assert mock._has_fit

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
def test_sample(sample_rate, ifos, dist):
    mock = MagicMock()
    mock_dist = MagicMock(side_effect=dist)

    mock.dec = mock_dist
    mock.phi = mock_dist
    mock.psi = mock_dist

    # first test sampling without
    # intrinsic parameters
    mock.intrinsic_parameters = None

    # first make sure we enforce fitting
    # if sampling with snr reweighting
    mock.snr = mock_dist
    mock._has_fit = False
    with pytest.raises(TypeError) as exc:
        RandomWaveformInjection.sample(mock, 1)
    assert str(exc.value).endswith("WaveformSampler.fit first")

    # now set background as mock
    # to mimick having called .fit
    mock._has_fit = True

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
    assert params.shape == (len(idx), 4)  # ra, dec, psi, snr
    assert (result == summed[idx]).all()
    mock.dec.assert_called_with(len(idx))

    # now test when snr is None
    # it doesn't get returned
    mock.snr = None
    result, params = RandomWaveformInjection.sample(mock, idx)
    assert params.shape == (len(idx), 3)  # ra, dec, psi
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

    # now pass intrinsic parameters of waveform to mock,
    # and ensure sampling returns them as expected
    num_intrinsic = 5
    intrinsic_parameters = torch.column_stack(
        [
            torch.arange(1, mock.num_waveforms, 1) * i
            for i in range(num_intrinsic)
        ]
    )
    mock.intrinsic_parameters = intrinsic_parameters
    result, params = RandomWaveformInjection.sample(mock, idx)
    assert (params[:, :num_intrinsic] == (intrinsic_parameters[idx])).all()


@pytest.fixture(params=[0.5, 1])
def prob(request):
    return request.param


@pytest.fixture
def dist():
    def f(N):
        return torch.Tensor(np.random.normal(size=N))

    return f


@patch("ml4gw.gw.reweight_snrs", new=reweight_snrs)
def test_random_waveform_injection(prob, ifos, dist):
    waveforms = {i: torch.randn(100, 1024) for i in ["plus", "cross"]}

    waveform = waveforms["plus"] + waveforms["cross"]
    summed = torch.stack([waveform + i for i in range(len(ifos))], axis=1)

    dec = Mock(side_effect=dist)
    psi = Mock(side_effect=dist)
    phi = Mock(side_effect=dist)
    snr = Mock(side_effect=dist)

    sample_rate = 1024
    # enforce all polarizations have to be same length
    with pytest.raises(ValueError) as exc:
        wrong = {str(i): np.random.randn(i + 2, 1024) for i in range(2)}
        transform = RandomWaveformInjection(
            sample_rate, ifos, dec, psi, phi, snr, prob=prob, **wrong
        )
    assert str(exc.value).startswith("Polarization")

    # enforce prob is greater than 0
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            sample_rate, ifos, dec, psi, phi, snr, prob=0, **waveforms
        )
    assert str(exc.value).startswith("Injection probability")

    # same for leq 1
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            sample_rate, ifos, dec, psi, phi, snr, prob=1.2, **waveforms
        )
    assert str(exc.value).startswith("Injection probability")

    # enforce parameters has same length as waveforms
    with pytest.raises(ValueError) as exc:
        num_intrinsic = 5
        wrong_intrinsic = np.zeros((len(waveforms) - 1, num_intrinsic))
        transform = RandomWaveformInjection(
            sample_rate,
            ifos,
            dec,
            psi,
            phi,
            snr,
            intrinsic_parameters=wrong_intrinsic,
            **waveforms
        )
    assert str(exc.value).startswith("Waveform parameters")

    # now make it for real. Start with snr=None to make sure
    # we don't register a buffer for the background
    transform = RandomWaveformInjection(
        sample_rate, ifos, dec, psi, phi, snr=None, prob=prob, **waveforms
    )

    # should be no registerd parameters
    assert len(list(transform.parameters())) == 0

    # should have buffers for tensors, vertices
    assert len(list(transform.buffers())) == 2
    assert transform.background is None

    # now with regular SNR
    transform = RandomWaveformInjection(
        sample_rate, ifos, dec, psi, phi, snr, prob=prob, **waveforms
    )
    assert len(list(transform.parameters())) == 0
    assert len(list(transform.buffers())) == 3

    assert transform.num_waveforms == 100
    assert transform.df == 1
    assert transform.mask is None
    assert not transform._has_fit

    assert transform.tensors.shape == (len(ifos), 3, 3)
    assert transform.vertices.shape == (len(ifos), 3)
    assert transform.background.shape == (len(ifos), sample_rate // 2 + 1)

    # create some fake data to inject into
    X = torch.zeros((16, len(ifos), 256))

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
    transform._has_fit = True
    with rand_patch as rand_mock, perm_patch as perm_mock:
        with randint_patch as randint_mock:
            X_hat, indices, params = transform(X)

    # ensure all the expected calls happened
    rand_mock.assert_called_with(size=(16,))
    perm_mock.assert_called_with(100)
    randint_mock.assert_called_with(256, 512, size=(expected_count,))
    dec.assert_called_with(expected_count)

    # now verify that the data matches up: the first
    # `exepcted_count` rows should be injected, and
    # the rest should be all 0s
    assert (indices == torch.arange(0, expected_count, 1)).all().item()
    for i, x_row in enumerate(X_hat):
        if i >= expected_count:
            assert (x_row == 0).all().item()
        else:
            assert (x_row == summed[i, :, i : i + 256]).all().item()

    # now try things with one of the parameters as a tensor
    dec = np.random.randn(100)

    # first make sure that having a param
    # too short raises an exception
    with pytest.raises(ValueError) as exc:
        transform = RandomWaveformInjection(
            sample_rate, ifos, dec[:99], psi, phi, snr, prob=prob, **waveforms
        )
    assert str(exc.value).startswith("Source parameter 'dec' is not")

    # now verify that the tensor is added as a buffer
    transform = RandomWaveformInjection(
        sample_rate, ifos, dec, psi, phi, snr, prob=prob, **waveforms
    )
    assert len(list(transform.buffers())) == 4

    transform._has_fit = True
    with rand_patch as rand_mock, perm_patch as perm_mock:
        with randint_patch as randint_mock:
            X_hat, indices, params = transform(X)

    # TODO: how to make sure this works?
    # expected_arg = torch.Tensor(dec[:expected_count])
    # compute_observed_strain.assert_called_with(expected_arg)

    for i, x_row in enumerate(X_hat):
        if i >= expected_count:
            assert (x_row == 0).all().item()
        else:
            assert (x_row == 2 * summed[i, :, i : i + 256]).all().item()

    # now re-make transform passing intrinsic parameters
    num_intrinsic = 5
    intrinsic_parameters = torch.randn((100, num_intrinsic))

    dec = torch.arange(0, 100, 1) * 1
    psi = torch.arange(0, 100, 1) * 2
    phi = torch.arange(0, 100, 1) * 3

    transform = RandomWaveformInjection(
        sample_rate,
        ifos,
        dec,
        psi,
        phi,
        prob=prob,
        intrinsic_parameters=intrinsic_parameters,
        **waveforms
    )

    # tensors, vertices, instrinsic, 3 x extrinsic
    assert len(list(transform.buffers())) == 6

    mask = torch.arange(0, 0.9, 0.9 / 16)
    perm_patch = patch("torch.randperm", return_value=torch.arange(16))
    rand_patch = patch("torch.rand", return_value=mask)
    randint_patch = patch(
        "torch.randint", return_value=torch.arange(expected_count)
    )

    X = torch.zeros((16, len(ifos), 256))
    with rand_patch as rand_mock, perm_patch as perm_mock:
        with randint_patch as randint_mock:
            X_hat, indices, params = transform(X)

    # all indices should have injection
    assert len(indices) == expected_count

    # make sure we're sampling correct params
    # corresponding to waveforms
    expected_params = torch.column_stack(
        (
            intrinsic_parameters[:expected_count],
            dec[:expected_count],
            psi[:expected_count],
            phi[:expected_count],
        )
    )
    assert (params == expected_params).all()

    for i, x_row in enumerate(X_hat):
        if i >= expected_count:
            assert (x_row == 0).all().item()
        else:
            assert (x_row == summed[i, :, i : i + 256]).all().item()

    # test that mask gets added as buffer
    transform = RandomWaveformInjection(
        sample_rate, ifos, dec, psi, phi, prob=prob, highpass=10, **waveforms
    )
    # no intrinsic, but includes mask
    assert len(list(transform.buffers())) == 6

    # now test IO
    weights_io = BytesIO()
    torch.save(transform.state_dict(), weights_io)

    weights_io.seek(0)
    transform.load_state_dict(torch.load(weights_io))
