import pytest

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


class TestSnrRescaler:
    @pytest.fixture(params=[1, 2])
    def num_channels(self, request):
        return request.param

    @pytest.fixture(params=[128, 512])
    def sample_rate(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def waveform_duration(self, request):
        return request.param

    @pytest.fixture(params=[None, 32])
    def highpass(self, request):
        return request.param

    @pytest.fixture(params=[None, 64])
    def lowpass(self, request):
        return request.param

    @pytest.fixture
    def transform(
        self, num_channels, sample_rate, waveform_duration, highpass, lowpass
    ):
        return SnrRescaler(
            num_channels, sample_rate, waveform_duration, highpass, lowpass
        )

    def test_init(
        self,
        transform,
        num_channels,
        sample_rate,
        waveform_duration,
        highpass,
        lowpass,
    ):
        assert not transform.built

        num_freqs = waveform_duration * sample_rate // 2 + 1
        shape = (num_channels, num_freqs)
        assert transform.background.shape == shape
        assert (transform.background == 0).all().item()

        if highpass is not None and lowpass is not None:
            start = int(highpass * waveform_duration)
            stop = int(lowpass * waveform_duration)
            assert len(transform._buffers) == 3
            assert not (transform.highpass_mask[:start].any().item())
            assert not (transform.lowpass_mask[stop:].any().item())
            assert transform.lowpass_mask[start:stop].all().item()
        elif highpass is not None:
            idx = int(highpass * waveform_duration)
            assert len(transform._buffers) == 2
            assert not (transform.highpass_mask[:idx]).any().item()
            assert (transform.highpass_mask[idx:]).all().item()
        elif lowpass is not None:
            idx = int(lowpass * waveform_duration)
            assert len(transform._buffers) == 2
            assert not (transform.lowpass_mask[idx:]).any().item()
            assert (transform.lowpass_mask[:idx]).all().item()
        else:
            assert len(transform._buffers) == 1
            assert transform.highpass_mask is None
            assert transform.lowpass_mask is None
