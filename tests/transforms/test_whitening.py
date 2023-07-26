from unittest.mock import patch

import numpy as np
import pytest
import torch
from gwpy.timeseries import TimeSeries

from ml4gw.spectral import spectral_density
from ml4gw.transforms import FixedWhiten, Whiten


@pytest.fixture(params=[512, 1024])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def num_ifos(request):
    return request.param


@pytest.fixture(params=[0.5, 1])
def fduration(request):
    return request.param


@pytest.fixture(params=[2, 4])
def fftlength(request):
    return request.param


@pytest.fixture(params=[0, 20])
def highpass(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def test_whitening_transform(
    num_ifos, sample_rate, fduration, fftlength, highpass, dtype
):
    tform = Whiten(num_ifos, sample_rate, fduration, dtype=dtype)

    # make sure that trying to run the forward
    # call without fitting raises an error
    kernel_length = fduration * 2
    kernel_size = int(kernel_length * sample_rate)
    X = torch.randn(8, num_ifos, kernel_size).type(dtype)
    X = 1e-19 + 1e-20 * X
    with pytest.raises(ValueError) as exc_info:
        tform(X)
    assert str(exc_info.value).startswith("Must fit parameters")

    # fit to some random background
    backgrounds = [np.random.randn(100 * sample_rate) for _ in range(num_ifos)]
    backgrounds = [1e-19 + 1e-20 * x for x in backgrounds]

    # verify that if our kernel length isn't greater
    # than the length of fduration we raise an error
    with pytest.raises(ValueError) as exc_info:
        tform.fit(
            fduration, *backgrounds, fftlength=fftlength, highpass=highpass
        )
    assert str(exc_info.value).startswith("Whitening pad size")

    # now fit to background
    tform.fit(
        kernel_length, *backgrounds, fftlength=fftlength, highpass=highpass
    )

    # make sure shape has to match
    with pytest.raises(ValueError) as exc_info:
        tform(X[:, :, :-1])
    assert f"kernel length of {kernel_length:0.1f}s" in str(exc_info.value)

    # apply transform
    results = tform(X).cpu().numpy()

    # check to make sure that whitened data is 0 mean unit variance
    # TODO: there are better statistically-motivated values to use
    # here, but we have to make the bounds so loose they ultimately
    # lose most meaning
    mean = results.mean()
    std = results.std()
    assert np.isclose(mean, 0, atol=0.02)
    assert np.isclose(std, 1, atol=0.1)

    # pre-compute the background asds for comparison
    asds = []
    for x in backgrounds:
        x = TimeSeries(x, sample_rate=sample_rate)
        x = x.asd(fftlength, method="median", window="hann")
        asds.append(x)

    # for each ifo in each batch element, verify that whitening
    # with gwpy produces roughly the same result
    pad = int(sample_rate * fduration / 2)
    for x, y in zip(X.cpu().numpy(), results):
        for input, result, asd in zip(x, y, asds):
            ts = TimeSeries(input, sample_rate=sample_rate)
            ts = ts.whiten(
                asd=asd, fduration=fduration, highpass=highpass, window="hann"
            )
            ts = ts.value[pad:-pad]

            # TODO: this isn't the strongest check in the world,
            # but it's the closest I can get for now. Check that
            # at least 95% of the outputs are within 1% of the
            # value produced by gwpy
            close = np.isclose(result, ts, rtol=1e-2)
            assert close.mean() > 0.95


def test_whitening_save_and_load(dtype, tmp_path):
    tform = Whiten(2, 512, 1, dtype)
    h1 = torch.randn(512 * 10)
    l1 = torch.randn(512 * 10)
    tform.fit(2, h1, l1, fftlength=2)

    assert (tform.kernel_length == 2).all().item()
    assert tform.built
    value = tform.time_domain_filter

    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(tform.state_dict(), tmp_path / "weights.pt")

    tform = Whiten(2, 512, 1, dtype)
    assert not tform.built
    assert (tform.kernel_length == 0).all().item()

    tform.load_state_dict(torch.load(tmp_path / "weights.pt"))
    assert tform.built
    assert (tform.kernel_length == 2).all().item()
    assert (tform.time_domain_filter == value).all().item()

    # ensure that loading something with the wrong shape
    # causes an error to get raised
    tform = Whiten(2, 512, 0.5, dtype)
    with pytest.raises(RuntimeError):
        tform.load_state_dict(torch.load(tmp_path / "weights.pt"))

    # ensure that we didn't set the built flag
    # during this attempt to read bad weights
    assert not tform.built

    # make sure saving and loading as part of
    # a larger module works as expected
    nn = torch.nn.Sequential(tform, torch.nn.Linear(512, 64))
    torch.save(nn.state_dict(), tmp_path / "weights.pt")
    nn.load_state_dict(torch.load(tmp_path / "weights.pt"))
    assert tform.built


class TestFixedWhiten:
    sample_rate = 256
    psd_length = 128
    whiten_length = 64
    num_channels = 5

    mean = 2
    std = 5

    def get_transform(self):
        return FixedWhiten(
            self.num_channels, self.whiten_length, self.sample_rate
        )

    @pytest.fixture
    def transform(self):
        return self.get_transform()

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

    def test_init(self, transform, X):
        # ensure parameters have been initialized
        # with the right shapes and to 0 valuess
        assert transform.psd.shape == (self.num_channels, 8193)
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
        transform.fit(2, *background, fftlength=4, highpass=highpass)
        assert (transform.psd != 0).all().item()
        assert (transform.fduration == 2).item()

        # now whiten a dummy tensor and make sure
        # that the values come out as expected
        whitened = transform(X)
        validate_whitened(
            whitened, highpass, self.sample_rate, 1 / self.whiten_length
        )

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
                psd, transform.psd[i], rtol=1e-6, atol=0.0
            )

        # now do a fit with a more realistic
        # fftlength, fduration, and highpass
        psds = self.get_psds(background, 4)
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
