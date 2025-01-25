from typing import Union

import torch
from scipy.signal import iirfilter
from torchaudio.functional import filtfilt


class IIRFilter(torch.nn.Module):
    r"""
    IIR digital and analog filter design given order and critical points.
    Design an Nth-order digital or analog filter and apply it to a signal.
    Uses SciPy's `iirfilter` function to create the filter coefficients.

    The forward call of this module accepts a batch tensor of shape
    (n_waveforms, n_samples) and returns the filtered waveforms.

    Args:
        N:
            The order of the filter.
        Wn:
            A scalar or length-2 sequence giving the critical frequencies.
            For digital filters, Wn are in the same units as fs. By
            default, fs is 2 half-cycles/sample, so these are normalized
            from 0 to 1, where 1 is the Nyquist frequency. (Wn is thus in
            half-cycles / sample). For analog filters, Wn is an angular
            frequency (e.g., rad/s). When Wn is a length-2 sequence,`Wn[0]`
            must be less than `Wn[1]`.
        rp:
            For Chebyshev and elliptic filters, provides the maximum ripple in
            the passband. (dB)
        rs:
            For Chebyshev and elliptic filters, provides the minimum
            attenuation in the stop band. (dB)
        btype:
            The type of filter. Default is 'bandpass'.
        analog:
            When True, return an analog filter, otherwise a digital filter
            is returned.
        ftype:
            The type of IIR filter to design:

                - Butterworth   : 'butter'
                - Chebyshev I   : 'cheby1'
                - Chebyshev II  : 'cheby2'
                - Cauer/elliptic: 'ellip'
                - Bessel/Thomson: 'bessel's
        fs:
            The sampling frequency of the digital system.

    Returns:
        These parameters are stored as torch buffers.
        b, a:
            Numerator (`b`) and denominator (`a`) polynomials of the IIR
            filter. Only returned if ``output='ba'``.
        z, p, k:
            Zeros, poles, and system gain of the IIR filter transfer
            function.  Only returned if ``output='zpk'``.
        sos:
            Second-order sections representation of the IIR filter.
            Only returned if ``output='sos'``.
    """

    def __init__(
        self,
        N: int,
        Wn: Union[float, torch.Tensor],
        rs: Union[None, float, torch.Tensor] = None,
        rp: Union[None, float, torch.Tensor] = None,
        btype="band",
        analog=False,
        ftype="butter",
        fs=None,
    ) -> None:
        super().__init__()

        if isinstance(Wn, torch.Tensor):
            _Wn = Wn.numpy()
        if isinstance(rs, torch.Tensor):
            _rs = rs.numpy()
        if isinstance(rp, torch.Tensor):
            _rp = rp.numpy()

        b, a = iirfilter(
            N,
            _Wn,
            rs=_rs,
            rp=_rp,
            btype=btype,
            analog=analog,
            ftype=ftype,
            output="ba",
            fs=fs,
        )
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("a", torch.tensor(a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the filter to the input signal.

        Args:
            x:
                The input signal to be filtered.

        Returns:
            The filtered signal.
        """
        return filtfilt(x, self.a, self.b, clamp=False)
