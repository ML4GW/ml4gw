from typing import List, Tuple

import bilby
import numpy as np
import torch
from bilby.core.utils import speed_of_light
from torchtyping import TensorType

batch = num_ifos = polarizations = time = space = None  # noqa

ScalarTensor = TensorType["batch"]
VectorGeometry = TensorType["space", "batch"]
TensorGeometry = TensorType[
    "batch",
    "space",
    "space",
]

NetworkVertices = TensorType["num_ifos", 3]
NetworkDetectorTensors = TensorType["num_ifos", 3, 3]
DetectorResponses = TensorType["batch", "num_ifos", "time"]


def outer(x: VectorGeometry, y: VectorGeometry) -> TensorGeometry:
    """
    Compute the outer product of two batches of
    vectors, with the vector dimension coming first.
    Moves the batch dimension in front before returning.
    """
    return torch.einsum("ik,jk->kij", x, y)


def plus(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, m) - outer(n, n)


def cross(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, n) + outer(n, m)


def breathing(m: VectorGeometry, n: VectorGeometry) -> TensorGeometry:
    return outer(m, m) + outer(n, n)


polarization_funcs = {
    "plus": plus,
    "cross": cross,
    "breathing": breathing,
}


def compute_antenna_responses(
    theta: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    detector_tensors: NetworkDetectorTensors,
    modes: List[str],
) -> TensorType["batch", "polarizations", "num_ifos"]:
    # it's simpler to stack these tensors to produce
    # 3 x batch_size tensors, then deal with rearranging
    # the batch dimension later, so that's what we'll do
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
            # note that the polarization functions will
            # move the batch dimension in front, and so
            # each polarization tensor will be batch x 3 x 3
            polarization = polarization_funcs[mode](m, n)
        except KeyError:
            raise ValueError(f"No polarization mode {mode}")
        polarizations.append(polarization)

    # shape: num_polarizations x batch x 3 x 3
    polarization = torch.stack(polarizations)

    # compute the weight of each interferometer's response
    # to each polarization: batch x polarizations x ifos
    return torch.einsum("pbjk,ijk->bpi", polarization, detector_tensors)


def shift_projections(
    projections: DetectorResponses,
    sample_rate: float,
    theta: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    vertices: NetworkVertices,
) -> DetectorResponses:
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
    antenna_responses: TensorType["batch", "polarizations", "num_ifos"],
    **polarizations: TensorType["batch", "time"],
) -> DetectorResponses:
    waveforms = torch.stack(list(polarizations.values()))
    return torch.einsum("bpi,pbt->bit", antenna_responses, waveforms)


def project_raw_gw(
    sample_rate: float,
    dec: ScalarTensor,
    psi: ScalarTensor,
    phi: ScalarTensor,
    detector_tensors: NetworkDetectorTensors,
    detector_vertices: NetworkVertices,
    **polarizations: TensorType["batch", "time"],
) -> DetectorResponses:
    """
    Project raw gravitational waveforms of different
    polarizations to the detector responses of a network
    of interferometers using the specified sky location
    parameters of the sources of the waveforms.

    Args:
        sample_rate:
            Rate at which the polarization tensors have
            sampled their time axis in Hz
        dec:
            Declination angle in the sky of the source
            of each waveform in radians
        psi:
            Angle between each waveform source's natural
            polarization basis and its polarization in
            Earth's coordinate frame in radians
        phi:
            Angle between each waveform source's right ascension
            and Earth's right ascension in radians
        detector_tensors:
            Concatenation of 3x3 matrices describing the
            spatial orientation of each interferometer onto
            which to project the gravitational waveforms
        detector_vertices:
            Concatenation of length 3 vectors describing
            the spatial location of each interferometer
            in Earth's coordinate system
        polarizations:
            Raw waveforms for each set of sky parameters provided,
            representing different polarizations of the same
            waveform for each batch index
    Returns:
        Tensor representing the observed strain at each
        interferometer for each waveform provided, with shape
        `(num_waveforms, num_ifos, waveform_size)`, where
        `(num_waveforms, waveform_size)` is the shape of each
        polarization tensor provided.
    """

    # TODO: just use theta as the input parameter?
    theta = np.pi / 2 - dec
    antenna_responses = compute_antenna_responses(
        theta, psi, phi, detector_tensors, list(polarizations)
    )

    # now sum along each polarization to get the projected
    # waveform for each interferometer: batch x ifos x time
    projections = project_waveforms(antenna_responses, **polarizations)

    # finally we shift the response of each interferometer
    # to account for the relative arrival time of each wave
    # based on the detector's location relative to the center
    # of the Earth
    return shift_projections(
        projections, sample_rate, theta, psi, phi, detector_vertices
    )


def get_ifo_geometry(
    *ifos: str,
) -> Tuple[NetworkDetectorTensors, NetworkVertices]:
    """
    For a given list of interferometer names, retrieve and
    concatenate the associated detector tensors and vertices
    of those interferometers.

    Args:
        ifos: Names of the interferometers whose geometry to retrieve
    Returns:
        A concatenation of the detector tensors of each interferometer
        A concatenation of the vertices of each interferometer
    """

    tensors, vertices = [], []
    for ifo in ifos:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo)

        tensors.append(ifo.detector_tensor)
        vertices.append(ifo.vertex)

    tensors = np.stack(tensors)
    vertices = np.stack(vertices)
    return torch.Tensor(tensors), torch.Tensor(vertices)
