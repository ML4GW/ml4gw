from unittest.mock import patch

import lal
import lalsimulation
import pytest
import torch

from ml4gw import spectral
from ml4gw.spectral import spectral_density
from ml4gw.transforms import (
    FixedMinimumPhaseWhiten,
    FixedWhiten,
    MinimumPhaseWhiten,
    Whiten,
)


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


class MinimumPhaseWhitenTest:
    sample_rate = 256
    fduration = 1
    num_channels = 2

    @property
    def size(self):
        return int(self.fduration * self.sample_rate)

    def get_psds(self):
        n_fft = 4 * self.size
        omega = 2 * torch.pi * torch.arange(n_fft // 2 + 1) / n_fft
        psds = []
        for coefficient in (0.2, 0.8):
            response = 1 - coefficient * torch.exp(-1j * omega)
            psds.append((2 / self.sample_rate) / response.abs().square())
        return psds


class TestMinimumPhaseWhiten(MinimumPhaseWhitenTest):
    def get_transform(self):
        return MinimumPhaseWhiten(self.fduration, self.sample_rate)

    def test_crop_is_causal(self):
        transform = self.get_transform()
        num_freqs = 2 * self.size + 1
        psd = torch.full((num_freqs,), 2 / self.sample_rate)
        X = torch.randn(4, self.num_channels, 1024)

        uncropped = transform(X, psd, crop=False)
        whitened = transform(X, psd)

        assert uncropped.shape == X.shape
        assert whitened.shape == (4, self.num_channels, 1024 - self.size)
        torch.testing.assert_close(whitened, uncropped[..., self.size :])

    @pytest.mark.parametrize("psd_ndim", [2, 3])
    def test_psd_broadcasting(self, psd_ndim):
        transform = self.get_transform()
        X = torch.randn(4, self.num_channels, 1024)
        psd = torch.full((2 * self.size + 1,), 2 / self.sample_rate)
        expected = transform(X, psd, crop=False)

        psd = psd.repeat(self.num_channels, 1)
        if psd_ndim == 3:
            psd = psd.repeat(X.size(0), 1, 1)

        whitened = transform(X, psd, crop=False)

        torch.testing.assert_close(whitened, expected)

    @pytest.mark.parametrize(
        ("highpass", "lowpass"),
        [(None, None), (32, None), (None, 100), (32, 100)],
    )
    def test_matches_standard_frequency_response(self, highpass, lowpass):
        n_fft = 16 * self.size
        psd = torch.full(
            (n_fft // 2 + 1,),
            2 / self.sample_rate,
            dtype=torch.float64,
        )
        transform = MinimumPhaseWhiten(
            self.fduration,
            self.sample_rate,
            highpass,
            lowpass,
        )
        impulse = torch.zeros(1, 1, n_fft, dtype=torch.float64)
        impulse[..., 0] = 1

        kernel = transform(impulse, psd, crop=False)[0, 0]
        response = torch.fft.rfft(kernel).abs()
        truncated_psd = spectral.truncate_inverse_power_spectrum(
            psd[None, None],
            self.fduration,
            self.sample_rate,
            highpass,
            lowpass,
        )[0, 0]
        expected = truncated_psd.rsqrt() / self.sample_rate**0.5

        torch.testing.assert_close(
            response, expected, rtol=0, atol=3e-3, check_dtype=False
        )

        if highpass is not None or lowpass is not None:
            frequencies = torch.fft.rfftfreq(
                n_fft, 1 / self.sample_rate, dtype=torch.float64
            )
            passband = frequencies >= (highpass or 0) + 5
            passband &= frequencies <= (lowpass or self.sample_rate / 2) - 5
            stopband = torch.zeros_like(passband)
            if highpass is not None:
                stopband |= frequencies <= highpass - 5
            if lowpass is not None:
                stopband |= frequencies >= lowpass + 5

            assert torch.quantile((response[passband] - 1).abs(), 0.95) < 0.01
            assert response[stopband].max() < 1e-3

    def test_realistic_aligo_psd_at_2048_hz(self):
        sample_rate = 2048
        fduration = 2
        n_fft = 8 * sample_rate
        num_freqs = n_fft // 2 + 1
        df = sample_rate / n_fft
        psd = lal.CreateREAL8FrequencySeries(
            "psd", 0, 0, df, "s^-1", num_freqs
        )
        lalsimulation.SimNoisePSDaLIGOaLIGO140MpcT1800545(psd, 1)
        psd = torch.from_numpy(psd.data.data.copy())
        transform = MinimumPhaseWhiten(
            fduration,
            sample_rate,
            highpass=20,
            lowpass=896,
        )
        impulse = torch.zeros(1, 1, n_fft, dtype=torch.float64)
        impulse[..., 0] = 1

        kernel = transform(impulse, psd, crop=False)[0, 0]
        response = torch.fft.rfft(kernel, n=n_fft).abs().double()
        truncated_psd = spectral.truncate_inverse_power_spectrum(
            psd[None, None],
            fduration,
            sample_rate,
            highpass=20,
            lowpass=896,
        )[0, 0]
        expected = truncated_psd.rsqrt() / sample_rate**0.5
        frequencies = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
        passband = (frequencies >= 40) & (frequencies <= 876)
        relative_error = (
            response[passband] - expected[passband]
        ).abs() / expected[passband]

        assert kernel.dtype == torch.float32
        assert torch.isfinite(kernel).all()
        assert relative_error.median() < 2e-4
        assert torch.quantile(relative_error, 0.95) < 2e-3
        assert relative_error.max() < 5e-3

    def test_filter_is_causal(self):
        transform = self.get_transform()
        psd = self.get_psds()[0]

        impulse_index = 512
        X = torch.zeros(1, self.num_channels, 1024)
        X[..., impulse_index] = 1
        whitened = transform(X, psd, crop=False)

        torch.testing.assert_close(
            whitened[..., :impulse_index],
            torch.zeros_like(whitened[..., :impulse_index]),
            rtol=0,
            atol=1e-7,
        )
        assert torch.count_nonzero(whitened[..., impulse_index:]) > 0

    def test_validation(self):
        with pytest.raises(ValueError, match="at least two samples"):
            MinimumPhaseWhiten(1 / self.sample_rate, self.sample_rate)

        transform = self.get_transform()
        X = torch.randn(4, self.num_channels, 1024)
        psd = torch.ones(2 * self.size + 1)
        with pytest.raises(ValueError, match="expected input with shape"):
            transform(X[0], psd)
        with pytest.raises(ValueError, match="one, two, or three dimensions"):
            transform(X, psd.repeat(2, 2, 2, 1))
        with pytest.raises(ValueError, match="channels"):
            transform(X, psd.repeat(self.num_channels + 1, 1))
        with pytest.raises(ValueError, match="batch size"):
            transform(X, psd.repeat(X.size(0) + 1, self.num_channels, 1))
        with pytest.raises(ValueError, match="Not enough timeseries"):
            transform(X[..., : self.size - 1], psd)
        with pytest.raises(ValueError, match="at least two frequency bins"):
            transform(X, torch.ones(1))
        with pytest.raises(ValueError, match="floating-point tensor"):
            transform(X, torch.ones(2 * self.size + 1, dtype=torch.int64))

    def test_upsamples_short_psd_before_factorization(self):
        transform = self.get_transform()
        psd = torch.full((self.size // 2 + 1,), 2 / self.sample_rate)
        X = torch.randn(2, self.num_channels, 1024)

        whitened = transform(X, psd, crop=False)

        assert whitened.shape == X.shape


class TestFixedMinimumPhaseWhiten(MinimumPhaseWhitenTest):
    def get_transform(self, dtype=torch.float64):
        return FixedMinimumPhaseWhiten(
            self.num_channels,
            self.fduration,
            self.sample_rate,
            dtype=dtype,
        )

    def test_crop_and_dynamic_equivalence(self):
        transform = self.get_transform()
        num_freqs = 2 * self.size + 1
        psd = torch.full((num_freqs,), 2 / self.sample_rate)
        transform.fit(psd, psd)

        X = torch.randn(4, self.num_channels, 1024)
        uncropped = transform(X, crop=False)
        whitened = transform(X)

        assert uncropped.shape == X.shape
        assert whitened.shape == (4, self.num_channels, 1024 - self.size)
        torch.testing.assert_close(whitened, uncropped[..., self.size :])
        dynamic = MinimumPhaseWhiten(self.fduration, self.sample_rate)
        expected = dynamic(X, psd, crop=False)
        torch.testing.assert_close(uncropped, expected, check_dtype=False)

    def test_filter_is_causal(self):
        transform = self.get_transform()
        transform.fit(*self.get_psds())

        impulse_index = 512
        X = torch.zeros(1, self.num_channels, 1024)
        X[..., impulse_index] = 1
        whitened = transform(X, crop=False)

        torch.testing.assert_close(
            whitened[..., :impulse_index],
            torch.zeros_like(whitened[..., :impulse_index]),
            rtol=0,
            atol=1e-7,
        )
        assert torch.count_nonzero(whitened[..., impulse_index:]) > 0

    def test_whitens_first_order_colored_noise(self):
        transform = self.get_transform()
        coefficient = 0.8
        n_fft = 4 * self.size
        omega = 2 * torch.pi * torch.arange(n_fft // 2 + 1) / n_fft
        response = 1 - coefficient * torch.exp(-1j * omega)
        psd = (2 / self.sample_rate) / response.abs().square()
        transform.fit(psd, psd)

        generator = torch.Generator().manual_seed(1234)
        noise = torch.randn(4, self.num_channels, 1024, generator=generator)
        X = torch.zeros_like(noise)
        X[..., 0] = noise[..., 0]
        for i in range(1, X.size(-1)):
            X[..., i] = coefficient * X[..., i - 1] + noise[..., i]

        whitened = transform(X, crop=False)
        frequencies = torch.fft.rfftfreq(X.size(-1), 1 / self.sample_rate)
        band = frequencies <= 0.8 * (self.sample_rate / 2)
        actual = torch.fft.rfft(whitened)[..., band]
        expected = torch.fft.rfft(noise)[..., band]
        relative_error = torch.linalg.vector_norm(actual - expected)
        relative_error /= torch.linalg.vector_norm(expected)

        assert relative_error < 5e-3

    def test_fit_from_timeseries(self):
        transform = self.get_transform()
        generator = torch.Generator().manual_seed(1234)
        backgrounds = [
            torch.randn(
                4 * self.sample_rate,
                generator=generator,
                dtype=torch.float64,
            )
            for _ in range(self.num_channels)
        ]

        transform.fit(*backgrounds, fftlength=2)
        X = torch.randn(
            2,
            self.num_channels,
            127,
            generator=generator,
            dtype=torch.float64,
        )
        whitened = transform(X, crop=False)

        assert transform.built
        assert whitened.shape == X.shape
        assert torch.isfinite(transform.kernel).all()
        assert torch.isfinite(whitened).all()

    def test_streaming_matches_continuous_input(self):
        transform = self.get_transform()
        transform.fit(*self.get_psds())

        generator = torch.Generator().manual_seed(1234)
        X = torch.randn(
            1,
            self.num_channels,
            1024,
            generator=generator,
            dtype=torch.float64,
        )
        expected = transform(X, crop=False)

        split = 600
        history = transform.kernel.size(-1)
        first = transform(X[..., :split], crop=False)
        second_input = X[..., split - history :]
        second = transform(second_input)
        actual = torch.cat((first, second), dim=-1)

        assert not torch.allclose(transform.kernel[0], transform.kernel[1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fft_filter_matches_direct_convolution(self):
        transform = self.get_transform()
        transform.fit(*self.get_psds())
        X = torch.randn(3, self.num_channels, 521, dtype=torch.float64)

        actual = transform(X, crop=False)
        padded = torch.nn.functional.pad(X, (self.size - 1, 0))
        expected = torch.nn.functional.conv1d(
            padded,
            torch.flip(transform.kernel[:, None], dims=(-1,)),
            groups=self.num_channels,
        ).float()

        torch.testing.assert_close(actual, expected)

    def test_fit_preserves_resolved_psd_grid(self):
        transform = self.get_transform()
        psd = torch.ones(8 * self.size + 1, dtype=torch.float64)
        psd[257:265] = 100

        with patch(
            "ml4gw.transforms.whitening."
            "spectral.minimum_phase_whitening_filter",
            wraps=spectral.minimum_phase_whitening_filter,
        ) as factorize:
            transform.fit(psd, psd)

        fitted_psd = factorize.call_args.args[0]
        assert fitted_psd.size(-1) == psd.size(-1)

    def test_truncated_filter_matches_standard_resolved_line(self):
        transform = self.get_transform()
        n_fft = 16 * self.size
        frequencies = torch.fft.rfftfreq(
            n_fft, 1 / self.sample_rate, dtype=torch.float64
        )
        psd = (2 / self.sample_rate) * (
            1 + 100 * torch.exp(-0.5 * ((frequencies - 60) / 2) ** 2)
        )
        transform.fit(psd, psd)

        response = torch.fft.rfft(transform.kernel[0], n=n_fft)
        whitened_psd = response.abs().square() * psd * self.sample_rate / 2
        truncated_psd = spectral.truncate_inverse_power_spectrum(
            psd[None, None],
            self.fduration,
            self.sample_rate,
        )[0, 0]
        expected = psd / (2 * truncated_psd)
        band = (frequencies >= 5) & (frequencies <= 120)
        log_error = whitened_psd[band].log().abs()

        torch.testing.assert_close(
            whitened_psd,
            expected,
            rtol=1e-8,
            atol=1e-10,
        )
        assert log_error.median() < 1e-3

    def test_fit_bandpass_matches_dynamic_transform(self):
        highpass = 32
        lowpass = 100
        n_fft = 16 * self.size
        psd = torch.full(
            (n_fft // 2 + 1,),
            2 / self.sample_rate,
            dtype=torch.float64,
        )
        transform = self.get_transform()
        transform.fit(psd, psd, highpass=highpass, lowpass=lowpass)
        dynamic = MinimumPhaseWhiten(
            self.fduration,
            self.sample_rate,
            highpass,
            lowpass,
        )
        X = torch.randn(2, self.num_channels, 1024, dtype=torch.float64)

        actual = transform(X, crop=False)
        expected = dynamic(X, psd, crop=False)

        torch.testing.assert_close(actual, expected)

    def test_preserves_fitted_precision_internally(self):
        transform = self.get_transform(dtype=torch.float64)
        psd = torch.ones(2 * self.size + 1, dtype=torch.float64)
        transform.fit(psd, psd)

        X = torch.randn(2, self.num_channels, 512, dtype=torch.float32)
        with patch("torch.fft.rfft", wraps=torch.fft.rfft) as rfft:
            whitened = transform(X, crop=False)

        assert transform.kernel.dtype == torch.float64
        assert all(
            call.args[0].dtype == torch.float64 for call in rfft.call_args_list
        )
        assert whitened.dtype == torch.float32

    def test_validation_and_io(self, tmp_path):
        transform = self.get_transform()
        X = torch.randn(4, self.num_channels, 1024)

        with pytest.raises(ValueError, match="at least two samples"):
            FixedMinimumPhaseWhiten(1, 1 / self.sample_rate, self.sample_rate)

        with pytest.raises(ValueError, match="Must fit parameters"):
            transform(X)
        with pytest.raises(ValueError, match="Expected to fit whitening"):
            transform.fit(torch.ones(129))

        psd = torch.ones(2 * self.size + 1)
        transform.fit(psd, psd)
        with pytest.raises(ValueError, match="expected input with shape"):
            transform(X[:, :1])

        path = tmp_path / "minimum-phase-whiten.pt"
        torch.save(transform.state_dict(), path)
        fresh = self.get_transform()
        fresh.load_state_dict(torch.load(path))
        torch.testing.assert_close(fresh(X), transform(X))
