from unittest.mock import patch

import pytest
import torch

from ml4gw.transforms.waveforms import WaveformSampler


@pytest.fixture(params=[100, 1000])
def num_waveforms(request):
    return request.param


def test_waveform_sampler(num_waveforms):

    # test that instantiating with different numbers of waveforms
    # for each polarization raises an error
    wrong = {str(i): torch.randn(i + 2, 1024) for i in range(2)}
    with pytest.raises(ValueError) as exc:
        WaveformSampler(**wrong)
    assert str(exc.value).startswith("Polarization")

    # instantiate waveform sampler
    waveforms = {
        i: torch.randn(num_waveforms, 1024) for i in ["plus", "cross"]
    }
    waveform_sampler = WaveformSampler(**waveforms)
    assert waveform_sampler.num_waveforms == num_waveforms
    assert waveform_sampler.parameters is None

    # test that requesting more waveforms than are available raises an error
    with pytest.raises(ValueError) as exc:
        waveform_sampler(num_waveforms + 1)
    assert str(exc.value).startswith("Requested")

    # test that sampling produces expected output
    samples = waveform_sampler(num_waveforms)
    for key in ["plus", "cross"]:
        assert samples[key].shape == (num_waveforms, 1024)

    # test that intrinsic parameters are returned if provided
    num_parameters = 5
    intrinsic_parameters = torch.column_stack(
        [torch.arange(0, num_waveforms, 1) * i for i in range(num_parameters)]
    )
    waveform_sampler = WaveformSampler(
        **waveforms, parameters=intrinsic_parameters
    )
    randint_patch = patch("torch.randint", return_value=torch.arange(10))
    with randint_patch:
        samples, params = waveform_sampler(10)

    assert len(samples["plus"]) == len(samples["cross"]) == len(params) == 10
    assert (params == (intrinsic_parameters[:10, :])).all()

    # test that passing in the wrong number of parameters raises an error
    wrong = torch.column_stack(
        [
            torch.arange(0, num_waveforms + 1, 1) * i
            for i in range(num_parameters)
        ]
    )
    with pytest.raises(ValueError) as exc:
        waveform_sampler = WaveformSampler(parameters=wrong, **waveforms)
    assert str(exc.value).startswith("Waveform parameters")
