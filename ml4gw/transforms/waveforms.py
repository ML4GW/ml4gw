from typing import List, Optional

import torch

from ml4gw import gw


# TODO: should these live in ml4gw.waveforms submodule?
# TODO: what in here should be stored as buffers?
class WaveformSampler(torch.nn.Module):
    def __init__(
        self,
        parameters: Optional[torch.Tensor] = None,
        **polarizations: torch.Tensor,
    ):
        super().__init__()
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

        if parameters is not None and len(parameters) != num_waveforms:
            raise ValueError(
                "Waveform parameters has {} waveforms "
                "associated with it, expected {}".format(
                    len(parameters), num_waveforms
                )
            )
        self.num_waveforms = num_waveforms
        self.parameters = parameters

    def forward(self, N: int):
        # TODO: should we allow sampling with replacement?
        if N > self.num_waveforms:
            raise ValueError(
                "Requested {} waveforms, but only {} are available".format(
                    N, self.num_waveforms
                )
            )
        # TODO: do we still really want this behavior here when a
        # user can do this without instantiating a WaveformSampler?
        elif N == -1:
            idx = torch.arange(self.num_waveforms)
            N = self.num_waveforms
        else:
            idx = torch.randint(self.num_waveforms, size=(N,))

        waveforms = {k: v[idx] for k, v in self.polarizations.items()}
        if self.parameters is not None:
            return waveforms, self.parameters[idx]
        return waveforms


class WaveformProjector(torch.nn.Module):
    def __init__(self, ifos: List[str], sample_rate: float):
        super().__init__()
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.sample_rate = sample_rate
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

    def forward(
        self,
        dec: gw.ScalarTensor,
        psi: gw.ScalarTensor,
        phi: gw.ScalarTensor,
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
