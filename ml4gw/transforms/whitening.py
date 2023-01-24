"""
Whitening logic largely lifted from gwpy's whitening functionality:

https://github.com/gwpy/gwpy/blob/main/gwpy/timeseries/timeseries.py
"""

from typing import Any, Mapping, Optional

import numpy as np
import torch
from gwpy.signal.filter_design import fir_from_transfer

from ml4gw.spectral import Background, normalize_psd
from ml4gw.transforms.transform import FittableTransform


class _Conv1d(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, X: torch.Tensor, tdf: torch.Tensor):
        return torch.nn.functional.conv1d(
            X, tdf, groups=self.num_channels, padding="same"
        )


def _overlap_add_conv(
    X: torch.Tensor, tdf: torch.Tensor, conv_op: torch.nn.Module, nfft: int
) -> torch.Tensor:
    """
    TODO: Stand-in implementation until we
    implement efficiently using windowing
    rather than looping
    """
    conv = torch.zeros_like(X)
    pad = conv_op.pad
    kernel_size = X.shape[-1]

    # handle first chunk separately
    y0 = conv_op(X[:, :, :nfft], tdf)
    conv[:, :, : nfft - pad] = y0[:, :, : nfft - pad]

    # process chunks of length nstep
    k = nfft - pad
    nstep = nfft - 2 * pad
    while k < kernel_size - nfft + pad:
        xk = X[:, :, k - pad : k + nstep + pad]
        yk = conv_op(xk, tdf)[:, :, pad:-pad]
        conv[:, :, k : k + yk.size(-1) - 2 * pad] = yk
        k += nstep

    # handle last chunk separately
    yf = conv_op(X[:, :, -nfft:], tdf)
    conv[:, :, -nfft + pad :] = yf[:, :, -nfft + pad :]
    return conv


class Whitening(FittableTransform):
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        fduration: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Whiten time domain data to background

        Whitens time domain data by some background to
        set the power across all frequency bins roughly
        to 0. Background data is passed to its `fit` method
        to create a time domain filter which is convolved with
        input data at call time.

        Args:
            num_channels: The number of timeseries channels to whiten
            sample_rate:
                The rate at which data on which this transform
                will be called will be sampled
            fduration:
                The length of the time domain filter in seconds.
                `fduration / 2` seconds will be cropped from either
                side of data passed to this transform, so that the
                output length, in seconds, of input timeseries of
                length `kernel_length` seconds will be
                `kernel_length - fduration`
            dtype:
                The datatype desired for the time domain filter
        """

        super().__init__()
        self.num_channels = num_channels
        self.sample_rate = sample_rate

        # shape properties of transfrom are only
        # functions of the fduration
        self.crop_samples = int((fduration / 2) * self.sample_rate)
        self.ntaps = int(fduration * self.sample_rate)
        self.pad = int((self.ntaps - 1) / 2)

        # the op that will actually convolve incoming
        # kernels with the time domain filter
        self.conv_op = _Conv1d(num_channels)

        # initialize the time domain filter with 0s,
        # then fill it out later
        tdf = torch.zeros((num_channels, 1, self.ntaps - 1), dtype=dtype)
        self.register_buffer("time_domain_filter", tdf)

        # save this as a parameter since it's decided at fit time
        kernel_length = torch.zeros((1,))
        self.register_buffer("kernel_length", kernel_length)

        # set up a window that we won't save with state
        # since it doesn't depend on the data at all
        window = torch.zeros((self.ntaps,), dtype=dtype)
        window[:-1] = torch.hann_window(self.ntaps - 1)
        self.register_buffer("window", window, persistent=False)

        # this will be set at fit time and will help us
        # decide which convolution implementation to use
        self.nfft = None

    def _check_kernel_length(self, kernel_length: float):
        kernel_size = int(kernel_length * self.sample_rate)
        if kernel_size <= (2 * self.crop_samples):
            raise ValueError(
                "Whitening pad size {} is too long for "
                "input kernel of size {}".format(
                    2 * self.crop_samples, kernel_size
                )
            )
        elif (8 * (self.ntaps - 1)) < (kernel_size / 2):
            self.nfft = min(8 * (self.ntaps - 1), kernel_size)
            # TODO: remove this error once the above
            # implementation is optimized
            raise NotImplementedError(
                "An optimal torch implementation of whitening for short "
                "filter padding is not complete. Use a larger value of pad."
            )

    def fit(
        self,
        kernel_length: float,
        *backgrounds: Background,
        fftlength: Optional[float] = None,
        highpass: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ) -> None:
        """Compute a time domain filter from background

        Args:
            kernel_length:
                The length in seconds of the timeseries data
                on which this transform will be applied after
                fitting.
            fftlength:
                If background data is passed as timeseries data,
                the length of the FFT to use to compute the PSD
                of the background data, in seconds. If left as `None`,
                will default to `1 / kernel_length`.
            highpass:
                Cutoff high-pass frequency for the frequency response
                of the fit time domain filter, in Hz. If left as `None`,
                frequency response will be the inverse of the background
                PSD across all bins.
            sample_rate:
                If background ata is passed as timeseries data,
                the rate at which it is sampled. If left as `None`,
                will default to the sample rate of the data on which
                this transform is meant to be applied.
            *backgrounds:
                Background data to use to fit the whitening time
                domain filter, whose frequency response in each
                channel will be the inverse of the corresponding
                background data's PSD. Should be passed in the
                same order these channels are expected to fall
                along the channel dimension of the data this
                transform is called on. Can be passed either as
                time domain or frequency domain data.
        """

        if len(backgrounds) != self.num_channels:
            raise ValueError(
                "Expected to fit whitening transform on {} background "
                "timeseries, but was passed {}".format(
                    self.num_channels, len(backgrounds)
                )
            )

        self._check_kernel_length(kernel_length)
        df = 1 / kernel_length
        ncorner = int(highpass / df) if highpass else 0

        tdfs = []
        for x in backgrounds:
            psd = normalize_psd(
                x, df, self.sample_rate, sample_rate, fftlength
            )

            tdf = fir_from_transfer(
                1 / psd**0.5,
                ntaps=self.ntaps,
                window="hann",
                ncorner=ncorner,
            )
            tdfs.append(tdf[:-1])

        tdfs = np.stack(tdfs)[:, None]
        tdf = torch.tensor(tdfs, dtype=self.time_domain_filter.dtype)
        kernel_length = torch.tensor((kernel_length,))
        super().build(time_domain_filter=tdf, kernel_length=kernel_length)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        expected_dim = int(self.kernel_length.item() * self.sample_rate)
        if X.size(-1) != expected_dim:
            raise ValueError(
                "Whitening transform was fit using a kernel length "
                "of {}s, but was passed data of length {}s".format(
                    self.kernel_length.item(), X.size(-1) / self.sample_rate
                )
            )

        # do a constant detrend along the time axis,
        X = X - X.mean(axis=-1, keepdims=True)

        # apply our window
        X[:, :, : self.pad] *= self.window[: self.pad]
        X[:, :, -self.pad :] *= self.window[-self.pad :]

        # apply different convolution depending on
        # length of time domain filter
        if self.nfft is None:
            X = self.conv_op(X, self.time_domain_filter)
        else:
            X = _overlap_add_conv(
                X, self.time_domain_filter, self.conv_op, self.nfft
            )

        # crop the beginning and ending fduration / 2
        X = X[:, :, self.crop_samples : -self.crop_samples]

        # scale by sqrt(2 / sample_rate) for some inscrutable
        # signal processing reason beyond my understanding
        return X * (2 / self.sample_rate) ** 0.5

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        keys = super().load_state_dict(state_dict, strict)
        self._check_kernel_length(self.kernel_length.item())
        return keys
