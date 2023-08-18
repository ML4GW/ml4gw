"""
Several implementation details are derived from the scipy csd and welch
implementations. For more info, see

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

and

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html
"""

from typing import Optional, Union

import torch
from torchtyping import TensorType

from ml4gw import types

time = None


def median(x, axis):
    """
    Implements a median calculation that matches numpy's
    behavior for an even number of elements and includes
    the same bias correction used by scipy's implementation.
    see https://github.com/scipy/scipy/blob/main/scipy/signal/_spectral_py.py#L2066 # noqa
    """
    n = x.shape[axis]
    ii_2 = 2 * torch.arange(1.0, (n - 1) // 2 + 1)
    bias = 1 + torch.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)
    return torch.quantile(x, q=0.5, axis=axis) / bias


def _validate_shapes(
    x: torch.Tensor, nperseg: int, y: Optional[torch.Tensor] = None
) -> None:
    if x.shape[-1] < nperseg:
        raise ValueError(
            "Number of samples {} in input x is insufficient "
            "for number of fft samples {}".format(x.shape[-1], nperseg)
        )
    elif x.ndim > 3:
        raise ValueError(
            f"Can't compute spectral density on tensor with shape {x.shape}"
        )

    if y is None:
        return

    # acceptable combinations of shapes:
    # x: time, y: time
    # x: channel x time, y: time OR channel x time
    # x: batch x channel x time, y: batch x channel x time OR batch x time
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            "Time dimensions of x and y tensors must "
            "be the same, found {} and {}".format(x.shape[-1], y.shape[-1])
        )
    elif x.ndim == 1 and not y.ndim == 1:
        raise ValueError(
            "Can't compute cross spectral density of "
            "1D tensor x with {}D tensor y".format(y.ndim)
        )
    elif x.ndim > 1 and y.ndim == x.ndim:
        if not y.shape == x.shape:
            raise ValueError(
                "If x and y tensors have the same number "
                "of dimensions, shapes must fully match. "
                "Found shapes {} and {}".format(x.shape, y.shape)
            )
    elif x.ndim > 1 and y.ndim != (x.ndim - 1):
        raise ValueError(
            "Can't compute cross spectral density of "
            "tensors with shapes {} and {}".format(x.shape, y.shape)
        )
    elif x.ndim > 2 and y.shape[0] != x.shape[0]:
        raise ValueError(
            "If x is a 3D tensor and y is a 2D tensor, "
            "0th batch dimensions must match, but found "
            "values {} and {}".format(x.shape[0], y.shape[0])
        )


def fast_spectral_density(
    x: torch.Tensor,
    nperseg: int,
    nstride: int,
    window: torch.Tensor,
    scale: torch.Tensor,
    average: str = "median",
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the power spectral density of a multichannel
    timeseries or a batch of multichannel timeseries, or
    the cross power spectral density of two such timeseries.
    This implementation is non-exact for the two lowest
    frequency bins, since it implements centering using the
    mean for the entire timeseries rather than on a per-
    window basis. The benefit of this is a faster implementation,
    which might be beneficial for cases where the lowest
    frequency components are going to be discarded anyway.

    Args:
        x:
            The timeseries tensor whose power spectral density
            to compute, or for cross spectral density the
            timeseries whose fft will be conjugated. Can have
            shape either
            `(batch_size, num_channels, length * sample_rate)`
            or `(num_channels, length * sample_rate)`.
        nperseg:
            Number of samples included in each FFT window
        nstride:
            Stride between FFT windows
        window:
            Window array to multiply by each FFT window before
            FFT computation. Should have length `nperseg // 2 + 1`.
        scale:
            Scale factor to multiply the FFT'd data by, related to
            desired units for output tensor (e.g. letting this equal
            `1 / (sample_rate * (window**2).sum())` will give output
            units of density, $\\text{Hz}^-1$$.
        average:
            How to aggregate the contributions of each FFT window to
            the spectral density. Allowed options are `'mean'` and
            `'median'`.
        y:
            Timeseries tensor to compute cross spectral density
            with `x`. If left as `None`, `x`'s power spectral
            density will be returned. Otherwise, if `x` is 1D,
            `y` must also be 1D. If `x` is 2D, the assumption
            is that this represents a single multi-channel timeseries,
            and `y` must be either 2D or 1D. In the former case,
            the cross-spectral densities of each channel will be
            computed individually, so `y` must have the same shape as `x`.
            Otherwise, this will compute the CSD of each of `x`'s channels
            with `y`. If `x` is 3D, this will be assumed to be a batch
            of multi-channel timeseries. In this case, `y` can either
            be 3D, in which case each channel of each batch element will
            have its CSD calculated or 2D, which has two different options.
            If `y`'s 0th dimension matches `x`'s 0th dimension, it will
            be assumed that `y` represents a batch of 1D timeseries, and
            for each batch element this timeseries will have its CSD with
            each channel of the corresponding batch element of `x`
            calculated. Otherwise, it sill be assumed that `y` represents
            a single multi-channel timeseries, in which case each channel
            of `y` will have its CSD calculated with the corresponding
            channel in `x` across _all_ of `x`'s batch elements.
    Returns:
        Tensor of power spectral densities of `x` or its cross spectral
            density with the timeseries in `y`.
    """

    _validate_shapes(x, nperseg, y)

    if x.ndim > 2:
        # stft only works on 2D input, so roll the
        # channel dimension out along the batch
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        x = x.reshape(-1, x.shape[-1])

        if y is not None and y.ndim == 3:
            # do the same thing for y if necessary
            y = y.reshape(-1, x.shape[-1])
    else:
        # this is just a multichannel timeseries, so
        # we can ignore batch reshapes later
        batch_size = None

    x = x - x.mean(axis=-1, keepdims=True)
    fft = torch.stft(
        x,
        n_fft=nperseg,
        hop_length=nstride,
        window=window,
        normalized=False,
        center=False,
        return_complex=True,
    )

    other = fft
    if y is not None:
        y = y - y.mean(axis=-1, keepdims=True)
        y_fft = torch.stft(
            y,
            n_fft=nperseg,
            hop_length=nstride,
            window=window,
            normalized=False,
            center=False,
            return_complex=True,
        )
        if batch_size is not None and fft.shape[0] != y_fft.shape[0]:
            # x is batched but y is not batched, so expand x's 0th
            # dimension back out so that we can multiply these arrays
            nfreq = nperseg // 2 + 1
            fft = fft.reshape(batch_size, num_channels, nfreq, -1)
            y_fft = y_fft[:, None]
        other = y_fft

    fft = torch.conj(fft) * other
    if y is None:
        fft = fft.real

    # some overly complex logic to find the frequency dimension
    # to apply this inscrutable signal processing step of
    # multiplying the non-edge frequency bins by 2
    # TODO: move this after aggregation since it won't affect
    # output value, and we have one less axis to do it over?
    stop = None if nperseg % 2 else -1
    if x.ndim == 1:
        fft[1:stop] *= 2
    elif fft.ndim < 4:
        fft[:, 1:stop] *= 2
    else:
        fft[:, :, 1:stop] *= 2
    fft *= scale

    if average == "mean":
        fft = fft.mean(axis=-1)
    else:
        # if this is a cross spectral density, fft will
        # be complex and so we'll compute the median along
        # both axes
        if y is not None:
            real_median = median(fft.real, -1)
            imag_median = 1j * median(fft.imag, -1)
            fft = real_median + imag_median
        else:
            fft = median(fft, -1)

    if fft.ndim == 2 and batch_size is not None:
        # if we still haven't expanded the batch dimension
        # back out, do so now
        fft = fft.reshape(batch_size, num_channels, -1)
    return fft


def spectral_density(
    x: torch.Tensor,
    nperseg: int,
    nstride: int,
    window: torch.Tensor,
    scale: torch.Tensor,
    average: str = "median",
) -> torch.Tensor:
    """
    Compute the power spectral density of a multichannel
    timeseries or a batch of multichannel timeseries, or
    the cross power spectral density of two such timeseries.
    This implementation is exact for all frequency bins, but
    slower than the fast implementation.

    Args:
        x:
            The timeseries tensor whose power spectral density
            to compute, or for cross spectral density the
            timeseries whose fft will be conjugated. Can have
            shape either
            `(batch_size, num_channels, length * sample_rate)`
            or `(num_channels, length * sample_rate)`.
        nperseg:
            Number of samples included in each FFT window
        nstride:
            Stride between FFT windows
        window:
            Window array to multiply by each FFT window before
            FFT computation. Should have length `nperseg // 2 + 1`.
        scale:
            Scale factor to multiply the FFT'd data by, related to
            desired units for output tensor (e.g. letting this equal
            `1 / (sample_rate * (window**2).sum())` will give output
            units of density, $\\text{Hz}^-1$$.
        average:
            How to aggregate the contributions of each FFT window to
            the spectral density. Allowed options are `'mean'` and
            `'median'`.
    """

    _validate_shapes(x, nperseg)

    # for non-fast implementation, we need to unfold
    # the tensor along the time dimension ourselves
    # to detrend each segment individually, so start
    # by converting x to a 4D tensor so we can use
    # torch's Unfold op
    if x.ndim == 1:
        reshape = []
        x = x[None, None, None, :]
    elif x.ndim == 2:
        reshape = [len(x)]
        x = x[None, :, None, :]
    elif x.ndim == 3:
        reshape = list(x.shape[:-1])
        x = x[:, :, None, :]

    # calculate the number of segments and trim x along
    # the time dimensions so that we can unfold it exactly
    num_segments = (x.shape[-1] - nperseg) // nstride + 1
    stop = (num_segments - 1) * nstride + nperseg
    x = x[..., :stop]

    # unfold x into overlapping segments and detrend and window
    # each one individually before computing the rfft. Unfold
    # will produce a batch x (num_channels * num_segments) x nperseg
    # shaped tensor
    unfold_op = torch.nn.Unfold((1, num_segments), dilation=(1, nstride))
    x = unfold_op(x)
    x = x - x.mean(axis=-1, keepdims=True)
    x *= window

    # after the fft, we'll have a
    # batch x (num_channels * num_segments) x nfreq
    # sized tensor
    fft = torch.fft.rfft(x, axis=-1).abs() ** 2

    if nperseg % 2:
        fft[..., 1:] *= 2
    else:
        fft[..., 1:-1] *= 2
    fft *= scale

    # unfold the batch and channel dimensions back
    # out if there were any to begin with, putting
    # the segment dimension as the second to last
    reshape += [num_segments, -1]
    fft = fft.reshape(*reshape)

    if average == "mean":
        return fft.mean(axis=-2)
    else:
        return median(fft, -2)


def truncate_inverse_power_spectrum(
    psd: types.PSDTensor,
    fduration: Union[TensorType["time"], float],
    sample_rate: float,
    highpass: Optional[float] = None,
) -> types.PSDTensor:
    """
    Truncate the length of the time domain response
    of a whitening filter built using the specified
    `psd` so that it has maximum length `fduration`
    seconds. This is meant to mitigate the impact
    of sharp features in the background PSD causing
    time domain responses longer than the segments
    to which the whitening filter will be applied.

    Implementation details adapted from
    https://github.com/vivinousi/gw-detection-deep-learning/blob/203966cc2ee47c32c292be000fb009a16824b7d9/modules/whiten.py#L8  # noqa

    Args:
        psd:
            The one-sided power spectraul density used
            to construct a whitening filter.
        fduration:
            Desired length in seconds of the time domain
            response of a whitening filter built using
            this PSD, or a window of this length to taper
            the edges of the time domain response of the
            filter. If passed as a float, a Hann window
            of this length will be used.
        sample_rate:
            Rate at which the time domain data to which
            the whitening filter will be applied has been
            sampled.
        highpass:
            If specified, will zero out the frequency response
            of all frequencies below this value in Hz. If left
            as `None`, no highpass filtering will be applied.
    Returns:
        The PSD with its time domain response truncated
            to `fduration` and any highpassed frequencies
            tapered.
    """

    num_freqs = psd.size(-1)
    N = (num_freqs - 1) * 2

    # use the inverse of the ASD as the
    # impulse response function
    inv_asd = 1 / psd**0.5

    # zero our leading frequencies if we want the
    # filter to perform highpass filtering
    if highpass is not None:
        df = sample_rate / N
        idx = int(highpass / df)
        inv_asd[:, :, :idx] = 0

    if inv_asd.size(-1) % 2:
        inv_asd[:, :, -1] = 0

    # now convert to time domain representation
    q = torch.fft.irfft(inv_asd, n=N, norm="forward", dim=-1)

    # taper the edges of the TD filter
    if isinstance(fduration, torch.Tensor):
        pad = fduration.size(-1) // 2
        window = fduration
    else:
        pad = int(fduration * sample_rate / 2)
        window = torch.hann_window(2 * pad, dtype=torch.float64)
        window = window.to(q.device)

    # 0 out anything else between the tapering regions
    q[:, :, :pad] *= window[-pad:]
    q[:, :, -pad:] *= window[:pad]
    if 2 * pad < q.size(-1):
        q[:, :, pad : q.size(-1) - pad] = 0

    # convert back to the frequency domain
    # to build the desired PSD
    inv_asd = torch.fft.rfft(q, n=N, norm="forward", dim=-1)
    inv_psd = inv_asd * inv_asd.conj()
    psd = 1 / inv_psd.abs()
    return psd / 2


def normalize_by_psd(
    X: types.WaveformTensor,
    psd: types.PSDTensor,
    sample_rate: float,
    pad: int,
):
    # compute the FFT of the section we want to whiten
    # and divide it by the ASD of the background section.
    # If the ASD of any background bin hit inf, set the
    # corresponding bin to 0
    X = X - X.mean(-1, keepdims=True)
    X_tilde = torch.fft.rfft(X.double(), norm="forward", dim=-1)
    X_tilde = X_tilde / psd**0.5
    X_tilde[torch.isnan(X_tilde)] = 0

    # convert back to the time domain and normalize
    # TODO: what's this normalization factor?
    X = torch.fft.irfft(X_tilde, norm="forward", dim=-1)
    X = X.float() / sample_rate**0.5

    # slice off corrupted data at edges of kernel
    X = X[:, :, pad:-pad]
    return X


def whiten(
    X: types.WaveformTensor,
    psd: types.PSDTensor,
    fduration: Union[TensorType["time"], float],
    sample_rate: float,
    highpass: Optional[float] = None,
) -> types.WaveformTensor:
    """
    Whiten a batch of timeseries using the specified
    background one-sided power spectral densities (PSDs),
    modified to have the desired time domain response length
    `fduration` and possibly to highpass filter.

    Args:
        X: batch of multichannel timeseries to whiten
        psd:
            PSDs use to whiten the data. The frequency
            response of the whitening filter will be roughly
            the inverse of the square root of this PSD, ensuring
            that data from the same distribution will have
            approximately uniform power after whitening.
            If 2D, each batch element in `X` will be whitened
            using the same PSDs. If 3D, each batch element will
            be whitened by the PSDs contained along the 0th
            dimenion of `psd`, and so the first two dimensions
            of `X` and `psd` should match.
        fduration:
            Desired length in seconds of the time domain
            response of a whitening filter built using
            this PSD, or a window of this length to taper
            the edges of the time domain response of the
            filter. If passed as a float, a Hann window
            of this length will be used. Moreover, half of
            this length will be removed from each edge of
            the whitened timeseries to account for filter
            settle-in time.
        sample_rate:
            Rate at which the data in `X` has been sampled
        highpass:
            The frequency in Hz at which to highpass filter
            the data, setting the frequency response in the
            whitening filter to 0. If left as `None`, no
            highpass filtering will be applied.
    Returns:
        Batch of whitened multichannel timeseries with
            `fduration / 2` seconds trimmed from each side.
    """

    # figure out how much data we'll need to slice
    # off after whitening
    if isinstance(fduration, torch.Tensor):
        pad = fduration.size(-1) // 2
    else:
        pad = int(fduration * sample_rate / 2)

    N = X.size(-1)
    if N <= (2 * pad):
        raise ValueError(
            "Not enough timeseries samples {} for "
            "number of padded samples {}".format(N, 2 * pad)
        )

    # normalize the number of expected dimensions in the PSD
    while psd.ndim < 3:
        psd = psd[None]

    # possibly interpolate our PSD to match the number
    # of frequency bins we expect to get from X
    num_freqs = N // 2 + 1
    if psd.size(-1) != num_freqs:
        # TODO: does there need to be any rescaling to
        # keep the integral of the PSD constant?
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs,), mode="linear"
        )

    # truncate it to have the desired
    # time domain response length
    psd = truncate_inverse_power_spectrum(
        psd, fduration, sample_rate, highpass
    )

    return normalize_by_psd(X, psd, sample_rate, pad)
