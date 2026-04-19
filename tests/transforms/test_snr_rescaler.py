import pytest
import torch

from ml4gw.gw import compute_network_snr
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

    @pytest.fixture
    def background(self, num_channels, sample_rate):
        size = int(8 * sample_rate)
        return [torch.randn(size) for _ in range(num_channels)]

    @pytest.fixture
    def responses(self, num_channels, sample_rate, waveform_duration):
        size = int(waveform_duration * sample_rate)
        return torch.randn(4, num_channels, size)

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

    def test_fit(self, transform, background, num_channels):
        # wrong number of channels raises an error before any state changes
        with pytest.raises(ValueError, match="Expected to fit"):
            transform.fit(*background[:-1], fftlength=1.0)
        assert not transform.built

        # fitting from time-domain background populates the buffer
        transform.fit(*background, fftlength=1.0)
        assert transform.built
        assert (transform.background != 0).all()

    def test_forward(self, transform, background, responses, sample_rate):
        transform.fit(*background, fftlength=1.0)

        batch_size = responses.shape[0]
        target_snrs = torch.full((batch_size,), 10.0)

        rescaled, returned_snrs = transform(responses, target_snrs)

        assert rescaled.shape == responses.shape
        torch.testing.assert_close(returned_snrs, target_snrs)

        # rescaled responses should have SNRs matching the targets
        actual_snrs = compute_network_snr(
            rescaled,
            transform.background,
            sample_rate,
            transform.highpass_mask,
            transform.lowpass_mask,
        )
        torch.testing.assert_close(
            actual_snrs.float(), target_snrs, rtol=1e-4, atol=0
        )

        # without explicit targets the returned SNRs are a permutation
        # of the input SNRs
        input_snrs = compute_network_snr(
            responses,
            transform.background,
            sample_rate,
            transform.highpass_mask,
            transform.lowpass_mask,
        )
        rescaled_rand, returned_rand = transform(responses)
        torch.testing.assert_close(
            torch.sort(returned_rand)[0],
            torch.sort(input_snrs)[0],
            rtol=1e-5,
            atol=0,
        )
        actual_rand_snrs = compute_network_snr(
            rescaled_rand,
            transform.background,
            sample_rate,
            transform.highpass_mask,
            transform.lowpass_mask,
        )
        torch.testing.assert_close(
            actual_rand_snrs.float(), returned_rand.float(), rtol=1e-4, atol=0
        )
