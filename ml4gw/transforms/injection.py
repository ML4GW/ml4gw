from collections.abc import Callable
from typing import Callable as CallableType
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ml4gw import gw
from ml4gw.spectral import Background, normalize_psd
from ml4gw.transforms.transform import FittableTransform
from ml4gw.utils.slicing import sample_kernels

Distribution = CallableType[[int], np.ndarray]
SourceParameter = Union[np.ndarray, Distribution]


class RandomWaveformInjection(FittableTransform):
    def __init__(
        self,
        sample_rate: float,
        ifos: List[str],
        dec: SourceParameter,
        psi: SourceParameter,
        phi: SourceParameter,
        snr: Optional[SourceParameter] = None,
        intrinsic_parameters: Optional[np.ndarray] = None,
        highpass: Optional[float] = None,
        prob: float = 1.0,
        trigger_offset: float = 0,
        **polarizations: np.ndarray,
    ) -> None:
        """Randomly inject gravitational waveforms into time domain data

        Transform that uses a bank of gravitational waveform
        polarizations and source parameters to generate interferometer
        responses which are randomly injected into background timeseries
        data. The `forward` method returns the
        combined background and injections tensor,
        the indices at which these injections where made,
        and the parameters used to generate the injections.

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
            sample_rate:
                Rate at which data used at call-time will be sampled.
            ifos:
                Interferometers onto which polarizations will be projected.
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
                this can be specified. If left as `None`, no SNR reweighting
                will be performed.
            intrinsic_parameters:
                Tensor containing the intrinsic parameters
                used to produce the passed polarizations.
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

        # store ifo geometries
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        # make sure we have the same number of waveforms
        # for all the different polarizations
        num_waveforms = waveform_size = None
        self.polarizations = {}
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

            # don't register these as buffers since they could
            # be large and we don't necessarily want them on
            # the same device as everything else
            self.polarizations[polarization] = torch.Tensor(tensor)

        if intrinsic_parameters is not None:
            if len(intrinsic_parameters) != num_waveforms:
                raise ValueError(
                    "Waveform parameters has {} waveforms "
                    "associated with it, expected {}".format(
                        len(intrinsic_parameters), num_waveforms
                    )
                )
            self.register_buffer(
                "intrinsic_parameters",
                torch.Tensor(intrinsic_parameters),
                persistent=False,
            )
        else:
            self.intrinsic_parameters = None

        # confirm that the source parameters all either
        # are a callable or have a length equal to the
        # number of waveforms
        names = ["dec", "psi", "phi", "snr"]
        for name, param in zip(names, [dec, psi, phi, snr]):
            if not isinstance(param, Callable) and param is not None:
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
                self.register_buffer(
                    name, torch.Tensor(param), persistent=False
                )
            else:
                setattr(self, name, param)

        self.num_waveforms = num_waveforms
        self.sample_rate = sample_rate
        self.df = sample_rate / waveform_size

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

        if snr is not None:
            num_freqs = int(waveform_size // 2 + 1)
            buff = torch.zeros((len(ifos), num_freqs), dtype=torch.float64)
            self.register_buffer("background", buff)
        else:
            self.background = None
            self.built = True

    def to(self, device, waveforms: bool = False):
        super().to(device)
        if waveforms:
            for t in self.polarizations.values():
                t.to(device)
        return self

    def fit(
        self,
        *backgrounds: Background,
        sample_rate: Optional[float] = None,
        fftlength: float = 2,
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

        if self.snr is None:
            raise TypeError("Cannot fit to backgrounds if snr is None")

        psds = []
        for background in backgrounds:
            psd = normalize_psd(
                background, self.df, self.sample_rate, sample_rate, fftlength
            )
            psds.append(psd)

        # save background as psd
        background = torch.tensor(np.stack(psds), dtype=torch.float64)
        super().build(background=background)

    def _sample_source_param(
        self, param: SourceParameter, idx: gw.ScalarTensor, N: int
    ):
        """
        Sample one of our source parameters either by calling it
        with `N` if it was specified as a callable object, or by
        slicing from it using `idx` otherwise.
        """
        if isinstance(param, Callable):
            return param(N).to(self.tensors.device)
        else:
            return param[idx]

    def __call__(self, *args, **kwargs):
        """
        Override __call__ method to original Module call
        since we do the `built` check in `sample`, not `forward`
        """
        return torch.nn.Module.__call__(self, *args, **kwargs)

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
            device:
                Device to map sampled waveforms to before projection
        Returns:
            Interferometer responses for each interferometer passed
                to `RandomWaveformInjection.fit` using each of the
                sampled waveforms and source parameters.
            The sampled source parameters used to generate the
                responses: `dec`, `psi`, `phi`, and the sampled SNRs.
        """
        if self.snr is not None:
            self._check_built()

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
        elif N_or_idx.ndim != 1:
            raise ValueError(
                "Can't slice waveforms with index tensor with {} dims".format(
                    N_or_idx.ndim
                )
            )
        else:
            # we provided specific waveform indices that
            # we would like to project
            idx = N_or_idx
            N = len(idx)

        dec = self._sample_source_param(self.dec, idx, N)
        psi = self._sample_source_param(self.psi, idx, N)
        phi = self._sample_source_param(self.phi, idx, N)

        # sample a batch of waveforms and move
        # them to the appropriate device
        polarizations = {}
        for polarization, waveforms in self.polarizations.items():
            waveforms = waveforms[idx]
            polarizations[polarization] = waveforms.to(dec.device)

        ifo_responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )

        if self.snr is not None:
            target_snrs = self._sample_source_param(self.snr, idx, N)
            rescaled_responses = gw.reweight_snrs(
                ifo_responses,
                target_snrs,
                backgrounds=self.background,
                sample_rate=self.sample_rate,
                highpass=self.mask,
            )

            sampled_params = torch.column_stack((dec, psi, phi, target_snrs))
        else:
            sampled_params = torch.column_stack((dec, psi, phi))
            rescaled_responses = ifo_responses

        if self.intrinsic_parameters is not None:
            intrinsic_parameters = self.intrinsic_parameters[idx]
            sampled_params = torch.column_stack(
                [intrinsic_parameters, sampled_params]
            )

        return rescaled_responses, sampled_params

    def forward(
        self,
        X: gw.WaveformTensor,
    ) -> gw.WaveformTensor:
        """Sample waveforms and inject them into random batch elements

        Batch elements from `X` will be selected at random for
        injection. Returns the tensor `X` with random injections,
        the indices where injections were done, and the parameters
        of the injections.
        """
        if self.training:
            mask = torch.rand(size=X.shape[:1]) < self.prob
            N = mask.sum().item()

            if N == 0:
                # return empty parameters if our roll
                # of the dice didn't turn up any waveforms
                param_shape = 3
                if self.snr is not None:
                    param_shape += 1
                if self.intrinsic_parameters is not None:
                    param_shape += self.intrinsic_parameters.size(-1)

                indices = torch.zeros((0,), device=X.device)
                sampled_params = torch.zeros((0, param_shape), device=X.device)
                return X, indices, sampled_params

            waveforms, sampled_params = self.sample(N)
            waveforms = sample_kernels(
                waveforms,
                kernel_size=X.shape[-1],
                max_center_offset=self.trigger_offset,
                coincident=True,
            )

            # map waveforms to appropriate device and
            # inject them into input tensor
            waveforms = waveforms.to(X.device)
            X[mask] += waveforms

            # make sure all our returns live on the same device
            indices = torch.where(mask)[0].to(X.device)
            sampled_params = sampled_params.to(X.device)
        else:
            # if we're in eval mode, skip injection
            # altogether and return nones to indicate this
            indices = sampled_params = None

        return X, indices, sampled_params
