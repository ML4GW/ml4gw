from typing import Callable, List, Optional

import torch

from ml4gw import gw
from ml4gw.distributions import ParameterSampler
from ml4gw.transforms.snr_rescaler import SnrRescaler
from ml4gw.utils.slicing import sample_kernels


class WaveformSampler(torch.nn.Module):
    def __init__(
        self,
        parameters: Optional[torch.Tensor] = None,
        **polarizations: torch.Tensor,
    ):
        # make sure we have the same number of waveforms
        # for all the different polarizations
        num_waveforms = None
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
                num_waveforms = tensor.shape[0]

            self.polarizations[polarization] = torch.Tensor(tensor)

        self.num_waveforms = num_waveforms
        self.parameters = parameters

    def foward(self, N: int):
        idx = torch.randint(self.num_waveforms, size=(N,))
        waveforms = {k: v[idx] for k, v in self.polarizations}
        if self.parameters is not None:
            return waveforms, self.parameters[idx]
        return waveforms


class WaveformProjector(torch.nn.Module):
    def __init__(self, *ifos: str, sample_rate: float):
        super().__init__()
        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        self.sample_rate = sample_rate
        self.register_buffer("tensors", self.tensors)
        self.register_buffer("vertices", self.vertices)

    def forward(
        self,
        dec: gw.ScalarTensor,
        phi: gw.ScalarTensor,
        psi: gw.ScalarTensor,
        **polarizations,
    ):
        ifo_responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        return ifo_responses


class RandomWaveformInjector(torch.nn.Module):
    def __init__(
        self,
        ifos: List[str],
        sample_rate: float,
        dec: Callable,
        psi: Callable,
        phi: Callable,
        rescaler: Optional[SnrRescaler] = None,
        prob: float = 1.0,
        trigger_offset: float = 0.0,
        **polarizations: torch.Tensor,
    ):
        super().__init__()
        if not 0 < prob <= 1.0:
            raise ValueError(
                f"Injection probability must be between 0 and 1, got {prob}"
            )
        self.prob = prob
        self.trigger_offset = int(trigger_offset * sample_rate)
        self.extrinisic_sampler = ParameterSampler(dec, phi, psi)
        self.projector = WaveformProjector(*ifos, sample_rate=sample_rate)
        self.waveform_sampler = WaveformSampler(**polarizations)
        self.rescaler = rescaler

    def to(self, device, waveforms: bool = False):
        super().to(device)
        if waveforms:
            for t in self.polarizations.values():
                t.to(device)
        return self

    def forward(self, X: gw.WaveformTensor, y: gw.ScalarTensor):
        if not self.training:
            return X, y

        mask = torch.rand(size=X.shape[:1]) < self.prob
        N = mask.sum().item()

        if N == 0:
            return X, y

        dec, psi, phi = self.extrinisic_sampler(N)
        waveforms = self.waveform_sampler(N)
        responses = self.projector(dec, phi, psi, **waveforms)
        if self.rescaler is not None:
            responses = self.rescaler(responses)

        responses = sample_kernels(
            responses,
            kernel_size=X.shape[-1],
            max_center_offset=self.trigger_offset,
            coincident=True,
        )

        # map waveforms to appropriate device and
        # inject them into input tensor
        responses = responses.to(X.device)

        X[mask] += responses
        y[mask] = 1
        return X, y
