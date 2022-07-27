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
    dec: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
    ifo_geometry: torch.Tensor,
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
    projections = np.einsum("bpi,pbt->bit", ifo_responses, waveforms)

    # TODO: add in time shifts
    return projections


def get_ifo_geometry(*ifos):
    geometries = []
    for ifo in ifos:
        ifo = bilby.gw.detector.get_empty_interferometer(ifo)
        geometries.append(ifo.detector_tensor.flatten())
    return torch.Tensor(np.stack(geometries))
