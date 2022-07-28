import bilby
import numpy as np
import torch


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


def project_raw_gw(
    sample_rate: float,
    dec: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
    ifo_geometry: torch.Tensor,
    ifo_vertices: torch.Tensor,
    **polarizations: torch.Tensor,
):
    modes = list(polarizations)
    waveforms = torch.stack([polarizations[m] for m in modes])

    # TODO: just use theta as the input parameter?
    theta = np.pi / 2 - dec
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

        # flatten the tensor out to make the einsum easier
        # polarization = polarization.reshape(-1, 9)
        polarizations.append(polarization)

    # polarizations x batch x 3 x 3
    polarization = torch.stack(polarizations)

    # compute the weight of each interferometer's response
    # to each polarization: batch x polarizations x ifos
    ifo_responses = torch.einsum("pbjk,ijk->bpi", polarization, ifo_geometry)

    # now sum along each polarization to get the projected
    # waveform for each interferometer: batch x ifos x time
    projections = torch.einsum("bpi,pbt->bit", ifo_responses, waveforms)
    batch_size, num_ifos, waveform_size = projections.shape

    # now compute the shift each interferometer experiences
    # as a result of the delayed (or advanced) arrival time
    # of the wave to its position relative to the geocenter
    trigger_shift = waveform_size / (2 * sample_rate)
    omega = torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    )
    dt = torch.einsum("jb,ji->bi", omega, ifo_vertices) + trigger_shift
    dt = torch.round(dt).type(torch.int64)

    # rolling by gathering implementation taken from
    # https://stackoverflow.com/a/68641864
    # waveform_size x 1 x 1
    idx = torch.arange(waveform_size)[:, None, None]

    # waveform_size x batch x num_ifos
    idx = idx.repeat((1, batch_size, num_ifos))
    idx = idx.to(omega.device)

    idx -= dt
    idx %= waveform_size

    rolled = []
    for i in range(num_ifos):
        ifo = torch.gather(projections[:, i], 1, idx[:, :, i])
        rolled.append(ifo[:, None])
    return torch.cat(rolled, axis=1)


def get_ifo_geometry(*ifos):
    tensors, vertices = [], []
    for ifo in ifos:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo)

        tensors.append(ifo.detector_tensor)
        vertices.append(ifo.vertex)

    tensors = np.stack(tensors)
    vertices = np.stack(vertices)
    return torch.Tensor(tensors), torch.Tensor(vertices)
