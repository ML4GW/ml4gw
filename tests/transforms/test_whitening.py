from unittest.mock import patch

import pytest
import torch

from ml4gw.spectral import spectral_density
from ml4gw.transforms import FixedWhiten, Whiten


class WhitenModuleTest:
    sample_rate = 8192
    psd_length = 128
    whiten_length = 64
    num_channels = 5

    mean = 2
    std = 5

    @pytest.fixture
    def background(self):
        background_size = int(self.psd_length * self.sample_rate)
        x = [torch.randn(background_size) for _ in range(self.num_channels)]
        return [self.mean + self.std * i for i in x]

    @pytest.fixture
    def X(self):
        size = int(self.whiten_length * self.sample_rate)
        X = torch.randn(8, self.num_channels, size)
        return self.mean + self.std * X

    @pytest.fixture(params=[None, 32])
    def highpass(self, request):
        return request.param

    def get_psds(self, background, fftlength):
        nperseg = int(fftlength * self.sample_rate)
        window = torch.hann_window(nperseg)
        psds = []
        for bg in background:
            psd = spectral_density(
                bg,
                nperseg,
                nperseg // 2,
                window,
                scale=1 / (self.sample_rate * (window**2).sum()),
            )
            psds.append(psd)
        return psds


class TestWhiten(WhitenModuleTest):
    fduration = 1

    @pytest.fixture
    def transform(self, highpass):
        return Whiten(self.fduration, self.sample_rate, highpass)

    def test_init(self, transform):
        assert transform.window.size(0) == 8192

    def test_forward(
        self, transform, X, background, highpass, validate_whitened
    ):
        background = self.get_psds(background, 2)
        background = torch.stack(background)
        whitened = transform(X, background)
        assert whitened.shape == (
            8,
            5,
            X.size(-1) - self.fduration * self.sample_rate,
        )
        validate_whitened(
            whitened, highpass, self.sample_rate, 1 / self.whiten_length
        )


class TestFixedWhiten(WhitenModuleTest):
    def get_transform(self):
        return FixedWhiten(
            self.num_channels, self.whiten_length, self.sample_rate
        )

    @pytest.fixture
    def transform(self):
        return self.get_transform()

    def test_init(self, transform, X):
        # ensure parameters have been initialized
        # with the right shapes and to 0 valuess
        num_freqs = (8192 * 64) // 2 + 1
        assert transform.psd.shape == (self.num_channels, num_freqs)
        assert (transform.psd == 0).all().item()

        assert transform.fduration.shape == (1,)
        assert (transform.fduration == 0).item()

        # ensure that trying to call forward
        # before fitting raises an error
        assert not transform.built
        with pytest.raises(ValueError) as exc:
            transform(X)
        assert str(exc.value).startswith("Must fit parameters")

    def test_fit_time_domain(
        self, transform, background, X, highpass, validate_whitened
    ):
        # ensure that calling with the wrong number
        # of background channels raises an error
        with pytest.raises(ValueError) as exc:
            transform.fit(2, *background[:-1], fftlength=4)
        assert str(exc.value).startswith("Expected to fit whitening")

        # fit to the background and ensure that the
        # parameter values have been set to nonzero values
        transform.fit(2, *background, fftlength=2, highpass=highpass)
        assert (transform.psd != 0).all().item()
        assert (transform.fduration == 2).item()

        # now whiten a dummy tensor and make sure
        # that the values come out as expected
        whitened = transform(X)
        validate_whitened(
            whitened, highpass, self.sample_rate, 1 / self.whiten_length
        )

    def test_fit_freq_domain(
        self, transform, background, X, highpass, validate_whitened
    ):
        # first check if fftlength == self.whiten_length,
        # the fit psd should match the psds used to fit
        # almost exactly if we don't use inverse spectrum
        # truncation
        psds = self.get_psds(background, self.whiten_length)
        with patch(
            "ml4gw.transforms.whitening"
            ".spectral.truncate_inverse_power_spectrum",
            new=lambda x, _, __, ___: x,
        ):
            transform.fit(2, *psds)
        assert (transform.fduration == 2).item()
        for i, psd in enumerate(psds):
            torch.testing.assert_close(
                psd, transform.psd[i], rtol=1e-6, atol=0.0, check_dtype=False
            )

        # now do a fit with a more realistic
        # fftlength, fduration, and highpass
        psds = self.get_psds(background, 2)
        transform.fit(2, *psds, highpass=highpass)
        assert (transform.psd != 0).all().item()
        assert (transform.fduration == 2).item()

        # now whiten a dummy tensor and make sure
        # that the values come out as expected
        whitened = transform(X)
        validate_whitened(
            whitened, highpass, self.sample_rate, 1 / self.whiten_length
        )

    def test_io(self, transform, background, tmp_path):
        transform.fit(2, *background, fftlength=4)
        assert transform.built
        tmp_path.mkdir(exist_ok=True, parents=True)
        torch.save(transform.state_dict(), tmp_path / "whiten.pt")

        fresh = self.get_transform()
        assert not fresh.built
        assert (fresh.fduration == 0).item()
        state_dict = torch.load(tmp_path / "whiten.pt")
        fresh.load_state_dict(state_dict)
        assert fresh.built

        assert (fresh.fduration == 2).item()
        assert (fresh.psd == transform.psd).all().item()
