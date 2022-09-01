from collections.abc import Callable
from typing import Callable as CallableType
from typing import Optional, Union

import numpy as np
import torch
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from ml4gw import gw
from ml4gw.utils.slicing import sample_kernels

Distribution = CallableType[[int], np.ndarray]
SourceParameter = Union[np.ndarray, Distribution]


class RandomWaveformInjection(torch.nn.Module):
    def __init__(
        self,
        dec: SourceParameter,
        psi: SourceParameter,
        phi: SourceParameter,
        snr: SourceParameter,
        sample_rate: float,
        highpass: Optional[float] = None,
        prob: float = 1.0,
        trigger_offset: float = 0,
        **polarizations: np.ndarray
    ) -> None:
        super().__init__()

        if not 0 < prob <= 1.0:
            raise ValueError(
                "Injection probability must be between 0 and 1, "
                "got {}".format(prob)
            )
        self.prob = prob
        self.trigger_offset = int(trigger_offset * sample_rate)

        # make sure we have the same number of waveforms
        # for all the different polarizations
        num_waveforms = waveform_size = None
        self.polarizations = torch.nn.ParameterDict()
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

            self.polarizations[polarization] = torch.nn.Parameter(
                torch.Tensor(tensor), requires_grad=False
            )

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
                        "Source parameter '{}' has type {}, must either "
                        "be callable or a numpy array".format(
                            name, type(param)
                        )
                    )

                if length != num_waveforms:
                    raise ValueError(
                        "Source parameter '{}' is not callable but "
                        "has length {}, expected length {}".format(
                            name, length, num_waveforms
                        )
                    )
                param = torch.Tensor(param)
                param = torch.nn.Parameter(param, requires_grad=False)
            setattr(self, name, param)

        self.num_waveforms = num_waveforms
        self.sample_rate = sample_rate
        self.df = sample_rate / waveform_size

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.mask = freqs >= highpass
        else:
            self.mask = None

        # initialize a bunch of properties we're
        # going to have to fit later
        self.background = self.tensors = self.vertices = None

    def fit(
        self,
        sample_rate: Optional[float] = None,
        fftlength: float = 2,
        **backgrounds: Union[np.ndarray, TimeSeries, FrequencySeries]
    ) -> None:
        ifos = list(backgrounds)
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.tensors = torch.nn.Parameter(tensors, requires_grad=False)
        self.vertices = torch.nn.Parameter(vertices, requires_grad=False)

        sample_rate = sample_rate or self.sample_rate
        psds = []
        for ifo, background in backgrounds.items():
            if not isinstance(background, FrequencySeries):
                # this is not already a frequency series, so we'll
                # assume it's a timeseries of some sort and convert
                # it to frequency space via psd
                if not isinstance(background, TimeSeries):
                    # if it's also not a TimeSeries object, then we'll
                    # assume that it's a numpy array which is sampled
                    # at the specified sample rate
                    background = TimeSeries(background, dt=1 / sample_rate)

                if background.dt != (1 / self.sample_rate):
                    # if the passed timeseries or specified sample
                    # rate doesn't match our sample rate here, resample
                    # so that we have the correct number of frequency bins
                    background = background.resample(self.sample_rate)

                # now convert to frequency space
                background = background.psd(
                    fftlength, method="median", window="hann"
                )

            # since the FFT length used to compute this PSD
            # won't, in general, match the length of waveforms
            # we're sampling, we'll interpolate the frequencies
            # to the expected frequency resolution
            background = background.interpolate(self.df)

            # since this is presumably real data, there shouldn't be
            # any 0s in the PSD. Otherwise this will lead to NaN SNR
            # values at reweighting time. TODO: is this really a
            # constraint we want to enforce, or should we leave this
            # to the user to catch?
            if (background == 0).any():
                raise ValueError(
                    "Found 0 values in background PSD "
                    "for interferometer {}".format(ifo)
                )
            psds.append(background.value)

        # save background as psd
        background = torch.Tensor(np.stack(psds))
        self.background = torch.nn.Parameter(background, requires_grad=False)

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
            if not (0 < N_or_idx <= self.num_waveforms or N_or_idx == -1):
                # we asked for too many waveforms, can't return enough
                raise ValueError(
                    "Can't sample {} waveforms from WaveformSampler "
                    "with {} waveforms associated with it".format(
                        N_or_idx, self.num_waveforms
                    )
                )
            elif N_or_idx == -1:
                # we asked for all the waveforms we have
                # TODO: should this be a randperm?
                idx = torch.arange(self.num_waveforms)
                N = self.num_waveforms
            else:
                # we asked for some specific random number of waveforms
                idx = torch.randperm(self.num_waveforms)[:N_or_idx]
                N = N_or_idx
        else:
            # we provided specific waveform indices that
            # we would like to project
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
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations
        )

        target_snrs = self._sample_source_param(self.snr, idx, N)
        ifo_responses = gw.reweight_snrs(
            ifo_responses,
            target_snrs,
            backgrounds=self.background,
            sample_rate=self.sample_rate,
            highpass=self.mask,
        )

        sampled_params = (dec, psi, phi, target_snrs)
        return ifo_responses, sampled_params

    def forward(self, X: gw.WaveformTensor, y: gw.ScalarTensor):
        if not self.training:
            return X, y

        mask = torch.rand(size=X.shape[:1]) < self.prob
        N = mask.sum().item()
        waveforms, _ = self.sample(N)
        waveforms = sample_kernels(
            waveforms,
            kernel_size=X.shape[-1],
            max_center_offset=self.trigger_offset,
            coincident=True,
        )

        X[mask] += waveforms
        y[mask] = 1
        return X, y
