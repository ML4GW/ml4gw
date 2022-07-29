from typing import List

import bilby
import numpy as np
import torch
from bilby.core.utils import speed_of_light
from torchtyping import TensorType

batch = num_ifos = polarizations = time = None  # noqa
ScalarTensor = TensorType["batch"]


def outer(x, y):
    return torch.einsum("ik,jk->kij", x, y)


def plus(m, n):
    return outer(m, m) - outer(n, n)


def cross(m, n):
    return outer(m, n) + outer(n, m)


def breathing(m, n):
    return outer(m, m) + outer(n, n)


polarization_funcs = {
    "plus": plus,
    "cross": cross,
    "breathing": breathing,
}


def compute_ifo_responses(
    theta: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    detector_tensors: TensorType["num_ifos", 3, 3],
    modes: List[str],
) -> TensorType["batch", "polarizations", "num_ifos"]:
    u = torch.stack(
        [
            torch.cos(phi) * torch.cos(theta),
            torch.cos(theta) * torch.sin(phi),
            -torch.sin(theta),
        ]
    )
    v = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)])

    m = -u * torch.sin(psi) - v * torch.cos(psi)
    n = -u * torch.cos(psi) + v * torch.sin(psi)

    # compute the polarization tensor of each signal
    # as a function of its parameters
    polarizations = []
    for mode in modes:
        try:
            polarization = polarization_funcs[mode](m, n)
        except KeyError:
            raise ValueError(f"No polarization mode {mode}")
        polarizations.append(polarization)

    # polarizations x batch x 3 x 3
    polarization = torch.stack(polarizations)

    # compute the weight of each interferometer's response
    # to each polarization: batch x polarizations x ifos
    ifo_responses = torch.einsum(
        "pbjk,ijk->bpi", polarization, detector_tensors
    )
    return ifo_responses


def shift_projections(
    projections: TensorType["batch", "num_ifos", "time"],
    sample_rate: float,
    theta: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    vertices: TensorType["num_ifos", 3],
):
    batch_size, num_ifos, waveform_size = projections.shape
    omega = torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    )
    dt = -torch.einsum("jb,ij->bi", omega, vertices)
    dt *= sample_rate / speed_of_light
    dt += waveform_size / 2
    dt = torch.floor(dt).type(torch.int64)

    # rolling by gathering implementation taken from
    # https://stackoverflow.com/a/68641864
    # waveform_size x 1 x 1
    idx = torch.arange(waveform_size)[:, None, None]

    # waveform_size x batch x num_ifos
    idx = idx.repeat((1, batch_size, num_ifos))
    idx = idx.to(omega.device)

    idx -= dt
    idx %= waveform_size
    idx = idx.transpose(0, 1)

    rolled = []
    for i in range(num_ifos):
        ifo = torch.gather(projections[:, i], 1, idx[:, :, i])
        rolled.append(ifo[:, None])
    return torch.cat(rolled, axis=1)


def project_waveforms(
    ifo_responses: TensorType["batch", "polarizations", "num_ifos"],
    **polarizations: TensorType["batch", "time"],
) -> TensorType["batch", "num_ifos", "time"]:
    waveforms = torch.stack(list(polarizations.values()))
    return torch.einsum("bpi,pbt->bit", ifo_responses, waveforms)


def project_raw_gw(
    sample_rate: float,
    dec: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
    detector_tensors: torch.Tensor,
    ifo_vertices: torch.Tensor,
    **polarizations: torch.Tensor,
):
    # TODO: just use theta as the input parameter?s
    theta = np.pi / 2 - dec
    ifo_responses = compute_ifo_responses(
        theta, psi, phi, detector_tensors, list(polarizations)
    )

    # now sum along each polarization to get the projected
    # waveform for each interferometer: batch x ifos x time
    projections = project_waveforms(ifo_responses, **polarizations)
    return shift_projections(
        projections, sample_rate, theta, psi, phi, ifo_vertices
    )


def get_ifo_geometry(*ifos):
    tensors, vertices = [], []
    for ifo in ifos:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo)

        tensors.append(ifo.detector_tensor)
        vertices.append(ifo.vertex)

    tensors = np.stack(tensors)
    vertices = np.stack(vertices)
    return torch.Tensor(tensors), torch.Tensor(vertices)
