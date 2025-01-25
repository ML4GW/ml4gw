import numpy as np
import torch
import torchaudio
from torch import Tensor

from .constants import PI

r"""
    Heavily based on the scipy implementation of the butterworth filter
    https://github.com/scipy/scipy/blob/main/scipy/signal/_filter_design.py
"""


def _buttap(N: int) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = torch.tensor([])
    m = torch.arange(-N + 1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -torch.exp(1j * PI * m / (2 * N))
    k = torch.tensor(1.0)
    return z, p, k


def _relative_degree(z: Tensor, p: Tensor) -> int:
    r"""
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError(
            "Improper transfer function. "
            "Must have at least as many poles as zeros."
        )
    else:
        return degree


def _lp2lp_zpk(
    z: Tensor, p: Tensor, k: Tensor, wo: float | Tensor = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Args:
        z:
            Zeros of the analog filter transfer function.
        p:
            Poles of the analog filter transfer function.
        k:
            System gain of the analog filter transfer function.
        wo:
            Desired cutoff, as angular frequency (e.g., rad/s).
            Defaults to no change.

    Returns:
        z_lp:
            Zeros of the transformed low-pass filter transfer function.
        p_lp:
            Poles of the transformed low-pass filter transfer function.
        k_lp:
            System gain of the transformed low-pass filter transfer function.
    """
    z = torch.atleast_1d(z)
    p = torch.atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def _lp2hp_zpk(
    z: Tensor, p: Tensor, k: Tensor, wo: float | Tensor = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Args:
        z:
            Zeros of the analog filter transfer function.
        p:
            Poles of the analog filter transfer function.
        k:
            System gain of the analog filter transfer function.
        wo:
            Desired cutoff, as angular frequency (e.g., rad/s).
            Defaults to no change.

    Returns:
        z_hp:
            Zeros of the transformed high-pass filter transfer function.
        p_hp:
            Poles of the transformed high-pass filter transfer function.
        k_hp:
            System gain of the transformed high-pass filter transfer function.
    """
    z = torch.atleast_1d(z)
    p = torch.atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = torch.cat((z_hp, torch.zeros(degree)))

    # Cancel out gain change caused by inversion
    k_hp = k * torch.real(torch.prod(-z) / torch.prod(-p)).item()

    return z_hp, p_hp, k_hp


def _lp2bp_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: float | Tensor = 1.0,
    bw: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Args:
        z:
            Zeros of the analog filter transfer function.
        p:
            Poles of the analog filter transfer function.
        k:
            System gain of the analog filter transfer function.
        wo:
            Desired center frequency, as angular frequency (e.g., rad/s).
            Defaults to no change.
        bw:
            Desired bandwidth, as angular frequency (e.g., rad/s).
            Defaults to no change.

    Returns:
        z_bp:
            Zeros of the transformed band-pass filter transfer function.
        p_bp:
            Poles of the transformed band-pass filter transfer function.
        k_bp:
            System gain of the transformed band-pass filter transfer function.
    """
    z = torch.atleast_1d(z)
    p = torch.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw / 2
    p_lp = p * bw / 2

    # Square root needs to produce complex result, not NaN
    z_lp = torch.tensor(z_lp, dtype=torch.complex128)
    p_lp = torch.tensor(p_lp, dtype=torch.complex128)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = torch.concatenate(
        (
            z_lp + torch.sqrt(z_lp**2 - wo**2),
            z_lp - torch.sqrt(z_lp**2 - wo**2),
        )
    )
    p_bp = torch.concatenate(
        (
            p_lp + torch.sqrt(p_lp**2 - wo**2),
            p_lp - torch.sqrt(p_lp**2 - wo**2),
        )
    )

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = torch.cat((z_bp, torch.zeros(degree)))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def _lp2bs_zpk(
    z: Tensor,
    p: Tensor,
    k: Tensor,
    wo: float | Tensor = 1.0,
    bw: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Args:
        z:
            Zeros of the analog filter transfer function.
        p:
            Poles of the analog filter transfer function.
        k:
            System gain of the analog filter transfer function.
        wo:
            Desired center frequency, as angular frequency (e.g., rad/s).
            Defaults to no change.
        bw:
            Desired bandwidth, as angular frequency (e.g., rad/s).
            Defaults to no change.

    Returns:
        z_bs:
            Zeros of the transformed band-stop filter transfer function.
        p_bs:
            Poles of the transformed band-stop filter transfer function.
        k_bs:
            System gain of the transformed band-stop filter transfer function.
    """
    z = torch.atleast_1d(z)
    p = torch.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw / 2) / z
    p_hp = (bw / 2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = torch.tensor(z_hp, dtype=torch.complex128)
    p_hp = torch.tensor(p_hp, dtype=torch.complex128)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = torch.concatenate(
        (
            z_hp + torch.sqrt(z_hp**2 - wo**2),
            z_hp - torch.sqrt(z_hp**2 - wo**2),
        )
    )
    p_bs = torch.concatenate(
        (
            p_hp + torch.sqrt(p_hp**2 - wo**2),
            p_hp - torch.sqrt(p_hp**2 - wo**2),
        )
    )

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = torch.cat((z_bs, torch.full(degree, +1j * wo)))
    z_bs = torch.cat((z_bs, torch.full(degree, -1j * wo)))

    # Cancel out gain change caused by inversion
    k_bs = k * torch.real(torch.prod(-z) / torch.prod(-p)).item()

    return z_bs, p_bs, k_bs


def _validate_fs(fs, allow_none=True):
    r"""
    Check if the given sampling frequency is a scalar and raises an exception
    otherwise. If allow_none is False, also raises an exception for none
    sampling rates. Returns the sampling frequency as float or none if the
    input is none.
    """
    if fs is None:
        if not allow_none:
            raise ValueError("Sampling frequency can not be none.")
    else:  # should be float
        if _size(fs) != 1:
            raise ValueError("Sampling frequency fs must be a single scalar.")
        fs = float(fs)
    return fs


def _bilinear_zpk(
    z: Tensor, p: Tensor, k: Tensor, fs: float
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Args:
        z:
            Zeros of the analog filter transfer function.
        p:
            Poles of the analog filter transfer function.
        k:
            System gain of the analog filter transfer function.
        fs:
            Sampling frequency of the digital system.

    Returns:
        z_z:
            Zeros of the transformed digital filter transfer function.
        p_z:
            Poles of the transformed digital filter transfer function.
        k_z:
            System gain of the transformed digital filter transfer function
    """
    z = torch.atleast_1d(z)
    p = torch.atleast_1d(p)

    fs = _validate_fs(fs, allow_none=False)

    degree = _relative_degree(z, p)

    fs2 = 2.0 * fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = torch.cat((z_z, -torch.ones(degree)))

    # Compensate for gain change
    k_z = k * torch.real(torch.prod(fs2 - z) / torch.prod(fs2 - p)).item()

    return z_z, p_z, k_z


def _zpk2tf(z: Tensor, p: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Return polynomial transfer function representation from zeros and poles

    Args:
        z:
            Zeros of the transfer function.
        p:
            Poles of the transfer function.
        k:
            System gain.

    Returns:
        b:
            Numerator polynomial coefficients.
        a:
            Denominator polynomial coefficients.
    """
    z = torch.atleast_1d(z)
    k = torch.atleast_1d(k)
    if len(z.shape) > 1:
        temp = _poly(z[0])
        b = torch.empty((z.shape[0], z.shape[1] + 1), dtype=temp.dtype)
        if len(k) == 1:
            _k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = _k[i] * _poly(z[i])
    else:
        b = k * _poly(z)
    a = torch.atleast_1d(_poly(p))
    a, b = a.real, b.real
    return b, a


def _size(t):
    """
    Return the number of elements in the input tensor.
    """
    try:
        shape = t.shape
        if type(shape) == torch.Size:
            return torch.prod(torch.tensor(shape)).item()
        else:
            return np.prod(shape)
    except AttributeError:
        return 1


def _poly(seq_of_zeros: Tensor) -> Tensor:
    r"""
    Find the coefficients of a polynomial with given sequence of roots.
    Implemented from numpy's poly function.
    https://numpy.org/doc/stable/reference/generated/numpy.poly.html

    Args:
        seq:
            Sequence of roots of the polynomial.

    Returns:
        c:
            1D array of polynomial coefficients from highest to lowest degree:

            ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
            where c[0] always equals 1.
    """
    seq_of_zeros = torch.atleast_1d(seq_of_zeros)
    c = torch.tensor([1.0], dtype=torch.float64)
    if len(seq_of_zeros) == 0:
        return c
    else:
        c = torch.tensor([1.0, -seq_of_zeros[0]], dtype=torch.complex128)
        for s in seq_of_zeros[1:]:
            c = torchaudio.functional.convolve(
                c, torch.tensor([1.0, -s], dtype=torch.complex128)
            )
        return c


def _iirfilter(  # noqa: C901
    N: int,
    Wn: torch.Tensor,
    btype="band",
    analog=False,
    ftype="butter",
    output="ba",
    fs=None,
) -> tuple:
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = Wn / (fs / 2)

    if torch.any(Wn <= 0):
        raise ValueError("filter critical frequencies must be greater than 0")

    if _size(Wn) > 1 and not Wn[0] < Wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")

    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError(
            f"'{btype}' is an invalid bandtype for filter."
        ) from e

    try:
        typefunc = filter_dict[ftype]
    except KeyError as e:
        raise ValueError(f"'{ftype}' is not a valid basic IIR filter.") from e

    if output not in ["ba", "zpk", "sos"]:
        raise ValueError(f"'{output}' is not a valid output form.")

    if not analog:
        if torch.any(Wn <= 0) or torch.any(Wn >= 1):
            if fs is not None:
                raise ValueError(
                    "Digital filter critical frequencies must "
                    f"be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs/2})"
                )
            raise ValueError(
                "Digital filter critical frequencies " "must be 0 < Wn < 1"
            )
        fs = 2.0
        warped = 2 * fs * torch.tan(PI * Wn / fs)
    else:
        warped = Wn

    # Get analog lowpass prototype
    if typefunc == _buttap:
        z, p, k = typefunc(N)

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ("lowpass", "highpass", "low", "high", "l", "h"):
        if _size(Wn) != 1:
            raise ValueError(
                "Must specify a single critical frequency Wn "
                "for lowpass or highpass filter"
            )

        if btype in ("lowpass", "low", "l"):
            z, p, k = _lp2lp_zpk(z, p, k, wo=warped)
        elif btype in ("highpass", "high", "h"):
            z, p, k = _lp2hp_zpk(z, p, k, wo=warped)
    elif btype in (
        "bandpass",
        "bandstop",
        "band",
        "stop",
        "bp",
        "bs",
    ):
        try:
            bw = warped[1] - warped[0]
            wo = torch.sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError(
                "Wn must specify start and stop frequencies for "
                "bandpass or bandstop filter"
            ) from e
        if btype in ("bandpass", "bp", "band"):
            z, p, k = _lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype in ("bandstop", "bs", "stop"):
            z, p, k = _lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError(f"'{btype}' not implemented in _iirfilter.")

    # Find discrete equivalent if necessary
    if not analog and fs is not None:
        z, p, k = _bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (numer-denom)
    if output == "zpk":
        return z, p, k
    else:
        return _zpk2tf(z, p, k)


filter_dict = {
    "butter": _buttap,
    "butterworth": _buttap,
}

band_dict: dict = {
    "band": "bandpass",
    "bandpass": "bandpass",
    "pass": "bandpass",
    "bp": "bandpass",
    "bs": "bandstop",
    "bandstop": "bandstop",
    "bands": "bandstop",
    "stop": "bandstop",
    "l": "lowpass",
    "low": "lowpass",
    "lowpass": "lowpass",
    "lp": "lowpass",
    "high": "highpass",
    "highpass": "highpass",
    "h": "highpass",
    "hp": "highpass",
}
