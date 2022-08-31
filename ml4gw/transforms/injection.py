from collections.abc import Callable
from typing import Callable as CallableType
from typing import Optional, Union

import numpy as np
import torch
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from ml4gw import gw

# from torchtyping import TensorType


Distribution = CallableType[int, np.ndarray]
SourceParameter = Union[np.ndarray, Distribution]


class WaveformSampler(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        dec: SourceParameter,
        psi: SourceParameter,
        phi: SourceParameter,
        snr: SourceParameter,
        highpass: float = 0.0,
        **polarizations: np.ndarray
    ) -> None:
        super().__init__()

        # make sure we have the same number of waveforms
        # for all the different polarizations
        num_waveforms = waveform_size = None
        for polarization, tensor in polarizations.items():
            if num_waveforms is not None and len(tensor) != num_waveforms:
                raise ValueError(
                    "Polarization {} has {} waveforms "
                    "associated with it, expected {}".format(
                        polarization, len(tensor), num_waveforms
                    )
                )
            elif num_waveforms is None:
                num_waveforms, waveform_size = tensor.shape

        # confirm that the source parameters all either
        # are a callable or have a length equal to the
        # number of waveforms
        names = ["dec", "psi", "phi", "snr"]
        for name, param in zip(names, [dec, psi, phi, snr]):
            if not isinstance(param, Callable):
                try:
                    length = len(param)
                except AttributeError:
                    raise TypeError(
                        "Source parameter {} has type {}, must either "
                        "be callable or a numpy array".format(
                            name, type(param)
                        )
                    )

                if length != num_waveforms:
                    raise ValueError(
                        "Source parameter {} is not callable but "
                        "has length {}, expected length {}".format(
                            name, length, num_waveforms
                        )
                    )
                param = torch.Tensor(param)
            setattr(self, name, param)

        self.num_waveforms = num_waveforms
        self.sample_rate = sample_rate
        self.df = sample_rate / waveform_size

        highpass = highpass or 0
        freqs = torch.arange(waveform_size // 2 + 1) * self.df
        mask = freqs >= highpass
        self.mask = torch.Parameter(mask, requires_grad=False)

        self.background = self.ifos = self.tensors = self.vertices = None

    def fit(
        self,
        sample_rate: Optional[float] = None,
        **backgrounds: Union[np.ndarray, TimeSeries, FrequencySeries]
    ) -> None:
        self.ifos = list(backgrounds)
        tensors, vertices = gw.get_ifo_geometry(*self.ifos)
        self.tensors = torch.Parameter(tensors, requires_grad=False)
        self.vertices = torch.Parameter(vertices, requires_grad=False)

        sample_rate = sample_rate or self.sample_rate
        asds = []
        for ifo, background in backgrounds.items():
            if not isinstance(background, FrequencySeries):
                if not isinstance(background, TimeSeries):
                    background = TimeSeries(background, dt=1 / sample_rate)

                if background.dt != (1 / self.sample_rate):
                    background = background.resample(self.sample_rate)

                background = background.asd(
                    2, method="median", window="hanning"
                )

            background = background.interpolate(self.df)
            if (background == 0).any():
                raise ValueError(
                    "Found 0 values in background ASD "
                    "for interferometer {}".format(ifo)
                )
            asds.append(background.value)

        # save background as psd
        background = np.stack(asds) ** 2
        self.background = torch.Parameter(background, requires_grad=False)

    def _sample_source_param(
        self, param: SourceParameter, idx: gw.ScalarTensor, N: int
    ):
        if isinstance(param, Callable):
            return param(N)
        else:
            return param[idx]

    def sample(
        self, N_or_idx: Union[int, gw.ScalarTensor]
    ) -> gw.WaveformTensor:
        if self.background is None:
            raise TypeError(
                "WaveformSampler can't sample waveforms until "
                "it has been fit on detector background. Make sure "
                "to call WaveformSampler.fit first"
            )

        if not isinstance(N_or_idx, torch.Tensor):
            if not 0 < N_or_idx < self.num_waveforms or N_or_idx == -1:
                raise ValueError(
                    "Can't sample {} waveforms from WaveformSampler "
                    "with {} waveforms associated with it".format(
                        N_or_idx, self.num_waveforms
                    )
                )
            elif N_or_idx == -1:
                idx = torch.arange(self.num_waveforms)
            else:
                idx = torch.randperm(self.num_waveforms)[:N_or_idx]
            N = N_or_idx
        else:
            idx = N_or_idx
            N = len(idx)

        dec = self._sample_source_param(self.dec, idx, N)
        psi = self._sample_source_param(self.psi, idx, N)
        phi = self._sample_source_param(self.phi, idx, N)

        polarizations = {k: v[idx] for k, v in self.polarizations.items()}
        ifo_responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            self.tensors,
            self.vertices,
            self.sample_rate,
            **polarizations
        )

        target_snrs = self._sample_source_param(self.snr)
        ifo_responses = gw.reweight_snrs(
            ifo_responses,
            target_snrs,
            self.background,
            self.sample_rate,
            self.mask,
        )
        return ifo_responses
