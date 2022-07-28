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
    ifo_geometry = ifo_geometry.reshape(-1, 9)

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
        polarization = polarization.reshape(-1, 9)
        polarizations.append(polarization)

    # polarizations x batch x 9
    polarization = torch.stack(polarizations)

    # compute the weight of each interferometer's response
    # to each polarization: batch x polarizations x ifos
    ifo_responses = torch.einsum("pbj,ij->bpi", polarization, ifo_geometry)

    # now sum along each polarization to get the projected
    # waveform for each interferometer: batch x ifos x time
    projections = torch.einsum("bpi,pbt->bit", ifo_responses, waveforms)
    batch_size, num_ifos, waveform_size = projections.shape

    dt = waveform_size / (2 * sample_rate)
    omega = torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ]
    ).t()
    idx = torch.arange(waveform_size)[:, None].repeat((1, batch_size))
    idx = idx.to(omega.device)

    rolled = []
    for i in range(num_ifos):
        vertex = ifo_vertices[i]
        delay = dt + (omega * vertex).sum(axis=-1)
        delay = torch.round(delay).type(torch.int64)

        # rolling by gathering implementation taken from
        # https://stackoverflow.com/a/68641864
        gather_idx = (idx - delay) % waveform_size
        gather_idx = gather_idx.t()
        ifo = projections[:, i]
        ifo = torch.gather(ifo, 1, gather_idx)

        rolled.append(ifo[:, None])
    return torch.cat(rolled, axis=1)


def get_ifo_geometry(*ifos):
    tensors, vertices = [], []
    for ifo in ifos:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo)

        tensors.append(ifo.detector_tensor.flatten())
        vertices.append(ifo.vertex.flatten())

    tensors = np.stack(tensors)
    vertices = np.stack(vertices)
    return torch.Tensor(tensors), torch.Tensor(vertices)
