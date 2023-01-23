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
        """Torch module for performing whitening. The first and last
        (fduration / 2) seconds of data are corrupted by the whitening
        and will be cropped. Thus, the output length
        that is ultimately passed to the network will be
        (kernel_length - fduration)
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

        self._has_fit = False
        self._use_overlap_add = None
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
        fftlength: float = 2,
        highpass: Optional[float] = None,
        sample_rate: Optional[float] = None,
        **channels: Background,
    ) -> None:
        """
        Build a whitening time domain filter
        """
        if len(channels) != self.num_channels:
            raise ValueError(
                "Expected to fit whitening transform on {} background "
                "timeseries, but was passed {}".format(
                    self.num_channels, len(channels)
                )
            )

        self._check_kernel_length(kernel_length)
        df = 1 / kernel_length
        ncorner = int(highpass / df) if highpass else 0

        tdfs = np.zeros((self.num_channels, 1, self.ntaps - 1))
        for i, (channel, x) in enumerate(channels.items()):
            psd = normalize_psd(
                x, df, self.sample_rate, sample_rate, fftlength
            )
            if (psd == 0).any():
                raise ValueError(f"Found 0 values in {channel} background asd")

            tdf = fir_from_transfer(
                1 / psd**0.5,
                ntaps=self.ntaps,
                window="hann",
                ncorner=ncorner,
            )
            tdfs[i, 0] = tdf[:-1]

        tdf = torch.tensor(tdfs, dtype=self.time_domain_filter.dtype)
        kernel_length = torch.tensor((kernel_length,))
        super().build(time_domain_filter=tdf, kernel_length=kernel_length)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
