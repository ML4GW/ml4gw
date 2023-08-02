from typing import Optional

import torch

from ml4gw.spectral import spectral_density


class FittableTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.built = False

    def build(self, **params):
        state_dict = self.state_dict()
        for name, value in params.items():
            state_dict[name].copy_(value)
        self.built = True

    def _check_built(self):
        if not self.built:
            raise ValueError(
                "Must fit parameters of {} transform to data "
                "before calling forward step".format(self.__class__.__name__)
            )

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def _load_from_state_dict(self, *args):
        # keep track of number of error messages to see
        # if trying to load these weights causes another
        num_errs = len(args[-1])
        ret = super()._load_from_state_dict(*args)

        # if we didn't create any new errors, then
        # assume that we built correctly
        if len(args[-1]) == num_errs:
            self.built = True
        return ret


class FittableSpectralTransform(FittableTransform):
    def normalize_psd(
        self,
        x,
        sample_rate: float,
        num_freqs: int,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None,
    ):
        # if we specified an FFT length, convert
        # the (assumed) time-domain data to the
        # frequency domain
        if fftlength is not None:
            nperseg = int(fftlength * sample_rate)

            overlap = overlap or fftlength / 2
            nstride = nperseg - int(overlap * sample_rate)

            window = torch.hann_window(nperseg, dtype=torch.float64)
            scale = 1.0 / (sample_rate * (window**2).sum())
            x = spectral_density(
                x,
                nperseg=nperseg,
                nstride=nstride,
                window=window,
                scale=scale,
            )

        # add two dummy dimensions in case we need to inerpolate
        # the frequency dimension, since `interpolate` expects
        # a (batch, channel, spatial) formatted tensor as input
        x = x.view(1, 1, -1)
        if x.size(-1) != num_freqs:
            x = torch.nn.functional.interpolate(x, size=(num_freqs,))
        return x[0, 0]
