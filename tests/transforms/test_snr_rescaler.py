import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from ml4gw import gw
from ml4gw.transforms import SnrRescaler


@pytest.fixture(params=[128, 512])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2])
def waveform_duration(request):
    return request.param


@pytest.fixture(params=[["H1"], ["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.mark.parametrize("factor", [None, 1, 2, 0.5])
def test_snr_rescaler(sample_rate, ifos, waveform_duration, factor):
    background = [np.random.randn(2048) for _ in ifos]
    fit_sample_rate = sample_rate * factor if factor is not None else None
    n_ifos = len(ifos)

    scaler = SnrRescaler(n_ifos, sample_rate, waveform_duration)
    assert scaler.highpass is None
    assert scaler.mask is None
    assert (scaler.background == 0).all().item()

    # ensure calling forward method before fitting raises a ValueError
    with pytest.raises(ValueError) as exc:
        scaler(torch.zeros((n_ifos, waveform_duration * sample_rate)))
    assert str(exc.value).startswith("Must fit")

    scaler.fit(*background, sample_rate=fit_sample_rate)
    assert (scaler.background != 0).all().item()
    # Question for Alec: this assertion statement used to be
    # assert scaler.background.dtype == torch.float32,
    # but it appears .fit casts to float64. How was this passing before?
    assert scaler.background.dtype == torch.float64
    assert scaler.built

    # use this background as our target for passing things
    # other than numpy arrays
    target_background = scaler.background.cpu().numpy()

    # now try passing in TimeSeries objects with differing sample rates
    ts_sample_rate = fit_sample_rate or sample_rate
    background = [TimeSeries(x, dt=1 / ts_sample_rate) for x in background]

    scaler = SnrRescaler(n_ifos, sample_rate, waveform_duration)
    scaler.fit(*background)
    assert np.isclose(scaler.background, target_background, rtol=1e-6).all()

    # now try passing in pre-computed psds
    background = [
        x.resample(sample_rate).psd(2, method="median", window="hann")
        for x in background
    ]
    scaler = SnrRescaler(n_ifos, sample_rate, waveform_duration)
    scaler.fit(*background)
    assert np.isclose(scaler.background, target_background, rtol=1e-6).all()

    # now test that rescaling snrs without a passed distribution
    # results in a random permutation of the snrs
    waveforms = torch.randn((100, n_ifos, waveform_duration * sample_rate))
    snrs = gw.compute_network_snr(
        waveforms, scaler.background, sample_rate, scaler.mask
    )
    rescaled, _ = scaler(waveforms)
    rescaled_snrs = gw.compute_network_snr(
        rescaled, scaler.background, sample_rate, scaler.mask
    )
    assert np.isclose(sorted(snrs), sorted(rescaled_snrs), rtol=1e-3).all()

    # now test that rescaling snrs with a distribution
    # passed to the rescaler results in expected snrs
    def distribution(n):
        return torch.ones(n)

    scaler = SnrRescaler(
        n_ifos, sample_rate, waveform_duration, distribution=distribution
    )
    scaler.fit(*background)
    rescaled, _ = scaler(waveforms)
    rescaled_snrs = gw.compute_network_snr(
        rescaled, scaler.background, sample_rate, scaler.mask
    )
    assert np.isclose(sorted(rescaled_snrs), torch.ones(100), rtol=1e-6).all()
