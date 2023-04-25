from typing import Optional

import torch

from ml4gw import gw


# TODO: should these live in ml4gw.waveforms submodule?
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
