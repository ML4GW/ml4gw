import torch
from torchaudio.functional import filtfilt

from ..filters import _iirfilter


class IIRFilter(torch.nn.Module):
    r"""
    IIR digital and analog filter design given order and critical points.
    Design an Nth-order digital or analog filter and apply it to a signal.

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
        btype:
            The type of filter. Default is 'bandpass'.
        analog:
            When True, return an analog filter, otherwise a digital filter
            is returned.
        ftype:
            The type of IIR filter to design:

                - Buttersworth   : 'butter'
        output:
            Filter form of the output:

                - numerator/denominator (default)    : 'ba'
                - pole-zero                          : 'zpk'
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
    """

    def __init__(
        self,
        N: int,
        Wn: torch.Tensor,
        btype="band",
        analog=False,
        ftype="butter",
        output="ba",
        fs=None,
    ) -> None:
        super().__init__()
        b, a = _iirfilter(N, Wn, btype, analog, ftype, output, fs)
        self.register_buffer("b", b)
        self.register_buffer("a", a)

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
