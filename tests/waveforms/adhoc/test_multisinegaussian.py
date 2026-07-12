from collections import OrderedDict

import pytest
import torch

from ml4gw.waveforms import MultiSineGaussian, SineGaussian


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2.0, 4.0, 8.0])
def duration(request):
    return request.param


@pytest.fixture(params=[5, 10])
def n_max(request):
    return request.param


@pytest.fixture(params=[4, 16, 32])
def batch_size(request):
    return request.param


@pytest.fixture
def multi_sine_gaussian(sample_rate, duration, n_max):
    return MultiSineGaussian(
        sample_rate=sample_rate,
        duration=duration,
        n_max=n_max,
        max_shift=0.0,
    )


@pytest.fixture
def sine_gaussian(sample_rate, duration):
    return SineGaussian(sample_rate, duration)


@pytest.fixture
def init_parameters(batch_size, n_max):
    generator = torch.Generator().manual_seed(1234)
    parameters = OrderedDict()

    parameters["n_components"] = torch.randint(
        low=1,
        high=n_max + 1,
        size=(batch_size,),
        generator=generator,
    )

    for i in range(1, n_max + 1):
        parameters[f"hrss_{i}"] = torch.empty(
            batch_size, dtype=torch.float64
        ).uniform_(1.6e-23, 1.5e-22, generator=generator)
        parameters[f"quality_{i}"] = torch.empty(
            batch_size, dtype=torch.float64
        ).uniform_(3.0, 700.0, generator=generator)
        parameters[f"frequency_{i}"] = torch.empty(
            batch_size, dtype=torch.float64
        ).uniform_(30.0, 2048.0, generator=generator)
        parameters[f"phase_{i}"] = torch.empty(
            batch_size, dtype=torch.float64
        ).uniform_(0.0, torch.pi, generator=generator)
        parameters[f"eccentricity_{i}"] = torch.empty(
            batch_size, dtype=torch.float64
        ).uniform_(0.0, 1.0, generator=generator)

    return parameters


def test_init_parameters_shapes_and_bounds(init_parameters, batch_size, n_max):
    assert isinstance(init_parameters, OrderedDict)
    assert init_parameters["n_components"].shape == (batch_size,)
    assert torch.all(init_parameters["n_components"] >= 1)
    assert torch.all(init_parameters["n_components"] <= n_max)

    for i in range(1, n_max + 1):
        assert init_parameters[f"hrss_{i}"].shape == (batch_size,)
        assert init_parameters[f"quality_{i}"].shape == (batch_size,)
        assert init_parameters[f"frequency_{i}"].shape == (batch_size,)
        assert init_parameters[f"phase_{i}"].shape == (batch_size,)
        assert init_parameters[f"eccentricity_{i}"].shape == (batch_size,)

        assert torch.all(init_parameters[f"hrss_{i}"] >= 1.6e-23)
        assert torch.all(init_parameters[f"hrss_{i}"] <= 1.5e-22)
        assert torch.all(init_parameters[f"quality_{i}"] >= 3.0)
        assert torch.all(init_parameters[f"quality_{i}"] <= 700.0)
        assert torch.all(init_parameters[f"frequency_{i}"] >= 30.0)
        assert torch.all(init_parameters[f"frequency_{i}"] <= 2048.0)
        assert torch.all(init_parameters[f"phase_{i}"] >= 0.0)
        assert torch.all(init_parameters[f"phase_{i}"] <= torch.pi)
        assert torch.all(init_parameters[f"eccentricity_{i}"] >= 0.0)
        assert torch.all(init_parameters[f"eccentricity_{i}"] <= 1.0)


def test_multi_sine_gaussian_forward_matches_active_component_sum(
    sample_rate,
    duration,
    batch_size,
    multi_sine_gaussian,
    sine_gaussian,
    init_parameters,
):
    cross, plus = multi_sine_gaussian(**init_parameters)

    waveform_size = int(sample_rate * duration)
    expected_cross = torch.zeros(
        batch_size, waveform_size, dtype=torch.float64
    )
    expected_plus = torch.zeros(
        batch_size, waveform_size, dtype=torch.float64
    )

    for batch_idx in range(batch_size):
        n_components = init_parameters["n_components"][batch_idx].item()
        for component_idx in range(1, n_components + 1):
            component_cross, component_plus = sine_gaussian(
                quality=init_parameters[f"quality_{component_idx}"][
                    batch_idx
                ].reshape(1),
                frequency=init_parameters[f"frequency_{component_idx}"][
                    batch_idx
                ].reshape(1),
                hrss=init_parameters[f"hrss_{component_idx}"][batch_idx].reshape(
                    1
                ),
                phase=init_parameters[f"phase_{component_idx}"][
                    batch_idx
                ].reshape(1),
                eccentricity=init_parameters[f"eccentricity_{component_idx}"][
                    batch_idx
                ].reshape(1),
            )
            expected_cross[batch_idx] += component_cross.squeeze(0)
            expected_plus[batch_idx] += component_plus.squeeze(0)

    assert cross.shape == (batch_size, waveform_size)
    assert plus.shape == (batch_size, waveform_size)
    assert torch.allclose(cross, expected_cross)
    assert torch.allclose(plus, expected_plus)
