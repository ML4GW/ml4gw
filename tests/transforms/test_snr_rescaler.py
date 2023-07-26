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

    @pytest.fixture
    def transform(
        self, num_channels, sample_rate, waveform_duration, highpass
    ):
        return SnrRescaler(
            num_channels, sample_rate, waveform_duration, highpass
        )

    def test_init(
        self, transform, num_channels, sample_rate, waveform_duration, highpass
    ):
        assert not transform.built

        num_freqs = waveform_duration * sample_rate // 2 + 1
        shape = (num_channels, num_freqs)
        assert transform.background.shape == shape
        assert (transform.background == 0).all().item()

        if highpass is not None:
            idx = int(highpass * waveform_duration)
            assert len(transform._buffers) == 2
            assert not (transform.mask[:idx]).any().item()
            assert (transform.mask[idx:]).all().item()
        else:
            assert len(transform._buffers) == 1
            assert transform.mask is None
