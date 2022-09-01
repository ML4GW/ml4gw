from collections.abc import Callable
from typing import Callable as CallableType
from typing import Optional, Tuple, Union

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
        **polarizations: np.ndarray,
    ) -> None:
        """Randomly inject gravitational waveforms into time domain data

        Transform that uses a bank of gravitational waveform
        polarizations and source parameters to generate interferometer
        responses which are randomly injected into background timeseries
        data. If a target tensor is provided at call time, its value at
        all the batch elements on which an injection is performed is set to 1.

        Before this module can be used, it must be fit to the background
        PSDs of the interferometers whose responses to the raw gravitational
        waveforms will be calculated at call time. This ensures that the
        SNRs of the signals follows some desired distribution. To do this,
        call the `.fit` method with `**kwargs` mapping from the interferometer
        ID to the corresponding background data. For more information, see
        the documentation to `RandomWaveformInjection.fit`.

        Source parameters of the provided waveform polarizations, the
        arguments `dec`, `psi`, `phi`, and `snr` can be provided in one
        of two ways to achieve different behavior. If any of these
        arguments is a callable, sampling of that parameter at call time
        will be performed by calling it with a single argument specifying
        the desired number of samples. If any of these arguments is a
        torch `Tensor`, it must have the same length as the specified
        `polarizations`. At sampling time, this parameter will be sampled
        by slicing it using the same indices used to slice to raw waveform
        polarizations. This is intended to facilitate deterministic sampling
        for e.g. validation purposes. To just sample some waveforms and
        parameters, try using the `RandomWaveformInjection.sample` method
        on its own.

        Args:
            dec:
                Source parameter specifying the declination of each
                source in radians relative to the celestial north.
                See description above about how this can be specified.
            psi:
                Source parameter specifying the angle in radians
                between each source's natural polarization basis
                and the basis which has the 0th unit vector pointing
                along the celestial equator. See description above
                about how this can be specified.
            phi:
                Source parameter specifying the angle in radians between
                each source's right ascension and the right ascension of
                the geocenter. See description above about how this can
                specified.
            snr:
                Source parameter specifying the desired signal to noise
                ratio of each injection. See description above about how
                this can be specified.
            sample_rate:
                Rate at which data used at call-time will be sampled.
            highpass:
                Frequency below which PSD data will not contribute
                to the SNR calculation. If left as `None`, SNR will
                be calculated across all frequencies up to `sample_rate / 2`.
            prob:
                Likelihood with which each batch element sampled at
                call-time will have an injection performed on it.
            trigger_offset:
                Maximum distance from the center of each waveform that
                kernels for each injection will be sampled, in seconds.
                This assumes that the trigger time of each waveform is
                in the center of the polarization tensors. For example,
                the default value of 0 means that every sampled kernel will
                include the trigger time in its injection. A positive value
                like 1 indicates that the kernel can fall _at most_ 1 second
                before the trigger time. A negative value like -0.5 indicates
                that the trigger has to be in every kernel, safely nestled
                at least 0.5 seconds from the edge of the kernel.
            **polarizations:
                Tensors representing different polarizations of the
                raw time-domain gravitational waveforms that will be
                mapped to an interferometer response at call time. Each
                element along the 0th axis of these tensors should come
                from the same source as the corresponding element in the
                other tensors. Accordingly, these should all be the same
                size. Allowed values of polarization names are `"plus"`,
                `"cross"`, and `"breathing"`.
        """

        super().__init__()

        if not 0 < prob <= 1.0:
            raise ValueError(
                f"Injection probability must be between 0 and 1, got {prob}"
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
        **backgrounds: Union[np.ndarray, TimeSeries, FrequencySeries],
    ) -> None:
        """Fit the transform to a specific set of interferometer PSDs

        In order to ensure that injections follow a specified SNR
        distribution, it's necessary to provide a background PSD
        for each interferometer onto which injections are being
        performed. This function will calculate that background and
        retrieve the detector tensors and vertices of the specified
        interferometers for use at call-time. Importantly,
        interferometer backgrounds must be specified in the
        _order in which they'll fall along the channel dimension at
        call-time_.

        Args:
            sample_rate:
                The rate at which the background data has been
                sampled. Only necessary if the provided background
                is a numpy array, otherwise it will be ignored. If
                left as `None`, the sample rate provided at
                initialization will be used instead.
            fftlength:
                The window length to use when calculating the PSD
                from a background timeseries. Only necessary if
                the provided background is not already a
                `gwpy.frequencyseries.FrequencySeries`, and is
                ignored otherwise.
            **backgrounds:
                The background data for each interferometer whose
                response to calculate at call-time, mapping from the
                id of each interferometer to its background data.
                If background data is provided as a numpy array, it's
                assumed to be a timeseries with sample rate given by
                `sample_rate` (or `self.sample_rate` if this is `None`).
                If provided as a `gwpy.timeseries.TimeSeries`, it's
                resampled to `self.sample_rate` and turned into a PSD
                using the value of `fftlength`. Otherwise, if provided
                as a `gwpy.frequencyseries.FrequencySeries`, it's
                interpolated to a frequency resolution corresponding to
                the inverse of the length of `self.polarizations` in
                seconds.
        """

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
        """
        Sample one of our source parameters either by calling it
        with `N` if it was specified as a callable object, or by
        slicing from it using `idx` otherwise.
        """
        if isinstance(param, Callable):
            return param(N)
        else:
            return param[idx]

    def sample(
        self, N_or_idx: Union[int, gw.ScalarTensor]
    ) -> Tuple[gw.WaveformTensor, Tuple[gw.ScalarTensor, ...]]:
        """
        Sample some waveforms and source parameters and use them
        to compute interferometer responses. Returns both the
        sampled interferometer responses as well as the source
        parameters used to generate them.

        Args:
            N_or_idx:
                Either an integer specifying how many waveforms
                to sample randomly, or specific waveform indices
                to sample deterministically. If specified as `-1`,
                _all_ waveforms will be sampled in order.
        Returns:
            Interferometer responses for each interferometer passed
                to `RandomWaveformInjection.fit` using each of the
                sampled waveforms and source parameters.
            The sampled source parameters used to generate the
                responses: `dec`, `psi`, `phi`, and the sampled SNRs.
        """

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
            **polarizations,
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

    def forward(
        self, X: gw.WaveformTensor, y: Optional[gw.ScalarTensor] = None
    ) -> gw.WaveformTensor:
        """Sample waveforms and inject them into random batch elements

        Batch elements from `X` will be selected at random for
        injection. If `y` is specified, it will be assumed to
        be a target tensor and the corresponding rows of `y`
        will be set to `1`.
        """
        if self.training:
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

            if y is not None:
                y[mask] = 1

        if y is not None:
            return X, y
        return X
