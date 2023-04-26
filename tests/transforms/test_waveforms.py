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
