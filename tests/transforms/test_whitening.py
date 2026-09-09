from unittest.mock import patch

import pytest
import torch

from ml4gw.spectral import spectral_density
from ml4gw.transforms import FixedWhiten, MinimumPhaseWhiten, Whiten


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

    @pytest.fixture(params=[None, 512])
    def lowpass(self, request):
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
    def transform(self, highpass, lowpass):
        return Whiten(self.fduration, self.sample_rate, highpass, lowpass)

    def test_init(self, transform):
        assert transform.window.size(0) == 8192

    def test_forward(
        self, transform, X, background, highpass, lowpass, validate_whitened
    ):
        background = self.get_psds(background, 2)
        background = torch.stack(background)
        whitened = transform(X, background)
        filter_size = self.fduration * self.sample_rate
        assert whitened.shape == (8, 5, X.size(-1) - filter_size)
        validate_whitened(
            whitened,
            highpass,
            lowpass,
            self.sample_rate,
            1 / self.whiten_length,
        )

        # Check that not cropping produces the expected values and shape
        whitened_uncropped = transform(X, background, crop=False)
        pad = filter_size // 2
        assert torch.all(whitened_uncropped[..., pad:-pad] == whitened)


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
        self, transform, background, X, highpass, lowpass, validate_whitened
    ):
        # ensure that calling with the wrong number
        # of background channels raises an error
        with pytest.raises(ValueError) as exc:
            transform.fit(2, *background[:-1], fftlength=4)
        assert str(exc.value).startswith("Expected to fit whitening")

        # fit to the background and ensure that the
        # parameter values have been set to nonzero values
        transform.fit(
            2, *background, fftlength=2, highpass=highpass, lowpass=lowpass
        )
        assert (transform.psd != 0).all().item()
        assert (transform.fduration == 2).item()

        # Check that an error is raised for an unexpected input shape
        with pytest.raises(ValueError, match=r"Whitening transform expected*"):
            transform(X[..., :-1])
        # now whiten a dummy tensor and make sure
        # that the values come out as expected
        whitened = transform(X)
        validate_whitened(
            whitened,
            highpass,
            lowpass,
            self.sample_rate,
            1 / self.whiten_length,
        )

        # Check that not cropping produces the expected values and shape
        whitened_uncropped = transform(X, crop=False)
        pad = int(transform.fduration * transform.sample_rate) // 2
        assert torch.all(whitened_uncropped[..., pad:-pad] == whitened)

    def test_fit_freq_domain(
        self, transform, background, X, highpass, lowpass, validate_whitened
    ):
        # first check if fftlength == self.whiten_length,
        # the fit psd should match the psds used to fit
        # almost exactly if we don't use inverse spectrum
        # truncation
        psds = self.get_psds(background, self.whiten_length)
        with patch(
            "ml4gw.transforms.whitening.spectral.truncate_inverse_power_spectrum",
            new=lambda x, _, __, ___, ____: x,
        ):
            transform.fit(2, *psds)
        assert (transform.fduration == 2).item()
        for i, psd in enumerate(psds):
            torch.testing.assert_close(
                psd, transform.psd[i], rtol=1e-6, atol=0.0, check_dtype=False
            )

        # now do a fit with a more realistic
        # fftlength, fduration, highpass, and lowpass
        psds = self.get_psds(background, 2)
        transform.fit(2, *psds, highpass=highpass, lowpass=lowpass)
        assert (transform.psd != 0).all().item()
        assert (transform.fduration == 2).item()

        # now whiten a dummy tensor and make sure
        # that the values come out as expected
        whitened = transform(X)
        validate_whitened(
            whitened,
            highpass,
            lowpass,
            self.sample_rate,
            1 / self.whiten_length,
        )

        # Check that not cropping produces the expected values and shape
        whitened_uncropped = transform(X, crop=False)
        pad = int(transform.fduration * transform.sample_rate) // 2
        assert torch.all(whitened_uncropped[..., pad:-pad] == whitened)

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


class TestMinimumPhaseWhiten:
    sample_rate = 256
    kernel_length = 1
    num_channels = 2

    def get_transform(self):
        return MinimumPhaseWhiten(
            self.num_channels,
            self.kernel_length,
            self.sample_rate,
        )

    def test_flat_psd_is_identity(self):
        transform = self.get_transform()
        num_freqs = self.sample_rate // 2 + 1
        psd = torch.full((num_freqs,), 2 / self.sample_rate)
        transform.fit(psd, psd)

        X = torch.randn(4, self.num_channels, 1024)
        whitened = transform(X)

        assert whitened.shape == X.shape
        torch.testing.assert_close(whitened, X, rtol=1e-6, atol=1e-6)

    def test_filter_is_causal(self):
        transform = self.get_transform()
        frequencies = torch.linspace(0, 1, self.sample_rate // 2 + 1)
        psd = 1 + frequencies**2
        transform.fit(psd, psd)

        impulse_index = 512
        X = torch.zeros(1, self.num_channels, 1024)
        X[..., impulse_index] = 1
        whitened = transform(X)

        assert torch.count_nonzero(whitened[..., :impulse_index]) == 0
        assert torch.count_nonzero(whitened[..., impulse_index:]) > 0

    def test_whitens_first_order_colored_noise(self):
        transform = self.get_transform()
        coefficient = 0.8
        n_fft = int(self.kernel_length * self.sample_rate)
        omega = 2 * torch.pi * torch.arange(n_fft // 2 + 1) / n_fft
        response = 1 - coefficient * torch.exp(-1j * omega)
        psd = (2 / self.sample_rate) / response.abs().square()
        transform.fit(psd, psd)

        noise = torch.randn(4, self.num_channels, 1024)
        X = torch.zeros_like(noise)
        X[..., 0] = noise[..., 0]
        for i in range(1, X.size(-1)):
            X[..., i] = coefficient * X[..., i - 1] + noise[..., i]

        whitened = transform(X)
        torch.testing.assert_close(whitened, noise, rtol=1e-5, atol=1e-5)

    def test_validation_and_io(self, tmp_path):
        transform = self.get_transform()
        X = torch.randn(4, self.num_channels, 1024)

        with pytest.raises(ValueError, match="at least two samples"):
            MinimumPhaseWhiten(1, 1 / self.sample_rate, self.sample_rate)

        with pytest.raises(ValueError, match="Must fit parameters"):
            transform(X)
        with pytest.raises(ValueError, match="Expected to fit whitening"):
            transform.fit(torch.ones(129))

        psd = torch.ones(129)
        transform.fit(psd, psd)
        with pytest.raises(ValueError, match="expected input with shape"):
            transform(X[:, :1])

        path = tmp_path / "minimum-phase-whiten.pt"
        torch.save(transform.state_dict(), path)
        fresh = self.get_transform()
        fresh.load_state_dict(torch.load(path))
        torch.testing.assert_close(fresh(X), transform(X))
