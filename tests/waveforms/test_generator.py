from math import pi

import pytest
import torch

from ml4gw import distributions
from ml4gw.waveforms.generator import ParameterSampler, WaveformGenerator


@pytest.fixture(params=[10, 100, 1000])
def n_samples(request):
    return request.param


@pytest.fixture(params=[1, 2, 10])
def duration(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def sample_rate(request):
    return request.param


def test_parameter_sampler(n_samples):
    parameter_sampler = ParameterSampler(
        phi=torch.distributions.Uniform(0, 2 * pi),
        dec=distributions.Cosine(),
        snr=distributions.LogNormal(6, 4, 3),
    )

    samples = parameter_sampler(
        n_samples,
    )

    for k in ["phi", "dec", "snr"]:
        assert len(samples[k]) == n_samples


def test_waveform_generator(sample_rate, duration, n_samples):
    def waveform(amplitude, frequency, phase):
        frequency = frequency.view(-1, 1)
        amplitude = amplitude.view(-1, 1)
        phase = phase.view(-1, 1)

        strain = torch.arange(0, duration, 1 / sample_rate)
        hplus = amplitude * torch.sin(2 * pi * frequency * strain + phase)
        hcross = amplitude * torch.cos(2 * pi * frequency * strain + phase)

        hplus = hplus.unsqueeze(1)
        hcross = hcross.unsqueeze(1)

        waveforms = torch.cat([hplus, hcross], dim=1)
        return waveforms

    parameter_sampler = ParameterSampler(
        amplitude=torch.distributions.Uniform(0, 1),
        frequency=torch.distributions.Uniform(0, 1),
        phase=torch.distributions.Uniform(0, 2 * pi),
    )

    generator = WaveformGenerator(waveform, parameter_sampler)
    waveforms, parameters = generator(n_samples)

    for k in ["amplitude", "frequency", "phase"]:
        assert len(parameters[k]) == n_samples
    assert waveforms.shape == (n_samples, 2, duration * sample_rate)


def test_parameter_sampler_bilby_prior_equivalence():
    parameter_sampler = ParameterSampler(
        phi=distributions.Uniform(0, 2 * pi),
        dec=distributions.Cosine(),
        snr=distributions.PowerLaw(8, 20, -4),
        distance=distributions.PowerLaw(10, 100, 2),
    )
    bilby_equivalent = parameter_sampler.to_bilby_prior_dict()

    parameter_samples = parameter_sampler(100000)
    bilby_prior_samples = bilby_equivalent.sample(100000)
    for name, samples in parameter_samples.items():
        assert bilby_prior_samples[name].mean() == pytest.approx(
            samples.mean().numpy(), abs=1e-2, rel=1e-1
        )


def test_parameter_sampler_bilby_prior_raising(monkeypatch):
    import importlib

    import bilby

    monkeypatch.delattr(bilby, "prior")
    importlib.reload(distributions)
    with pytest.raises(RuntimeError):
        distributions.Cosine().bilby_prior_equivalent()
    with pytest.raises(RuntimeError):
        parameter_sampler = ParameterSampler(
            phi=distributions.Uniform(0, 2 * pi),
            dec=distributions.Cosine(),
            snr=distributions.PowerLaw(8, 20, -4),
            distance=distributions.PowerLaw(10, 100, 2),
        )
        parameter_sampler.to_bilby_prior_dict()
