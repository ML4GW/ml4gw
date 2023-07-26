import pytest
import torch


@pytest.fixture
def validate_whitened():
    def validate(whitened, highpass, sample_rate, df):
        # make sure we have 0 mean unit variance
        means = whitened.mean(axis=-1)
        target = torch.zeros_like(means)
        torch.testing.assert_close(means, target, rtol=0, atol=0.02)

        stds = whitened.std(axis=-1)
        target = torch.ones_like(stds)

        # if we're highpassing, then we shouldn't expect
        # the standard deviation to be one because we're
        # subtracting some power, so remove roughly the
        # expected power contributed by the highpassed
        # frequencies from the target
        if highpass is not None:
            nyquist = sample_rate / 2
            target *= (1 - highpass / nyquist) ** 0.5
        torch.testing.assert_close(stds, target, rtol=0.05, atol=0.0)

        # check that frequencies up to close to the highpass
        # frequency have near 0 power.
        # TODO: the tolerance will need to increase
        # the closer to the cutoff frequency we get,
        # so what's a better way to check this
        if highpass is not None:
            fft = torch.fft.rfft(whitened, norm="ortho").abs()
            idx = int(0.8 * highpass / df)
            passed = fft[:, :, :idx]
            target = torch.zeros_like(passed)
            torch.testing.assert_close(passed, target, rtol=0, atol=0.07)

    return validate
