import torch


def sample_chirp_distance(
    parameters: dict[str, torch.Tensor],
    reverse: bool = False,
    jacobian: bool = False,
) -> dict[str, torch.Tensor] | torch.Tensor:
    """
    Convert between chirp distance and luminosity distance using chirp-mass scaling.

    Parameters
    ----------
    parameters : dict[str, torch.Tensor]
        Dictionary containing parameter tensors. Must contain:
            - "chirp_mass": chirp mass tensor
            - "distance": distance tensor

        The "distance" entry is interpreted as:
            - chirp distance (Dc) when ``reverse=False``
            - luminosity distance (Dl) when ``reverse=True``

    reverse : bool, optional
        If ``False`` (default), convert chirp distance to luminosity distance:
            Dl = Dc * (Mc / M_ref)^(5/6)
        If ``True``, apply the inverse transformation:
            Dc = Dl / (Mc / M_ref)^(5/6)

    jacobian : bool, optional
        If ``True``, return the log-Jacobian term of the transformation
        instead of transformed parameters.

    Returns
    -------
    dict[str, torch.Tensor] or torch.Tensor
        If ``jacobian=False``, returns the modified parameter dictionary.

        If ``jacobian=True``, returns the log-Jacobian tensor:

            -log((Mc / M_ref)^(5/6))
    """
    m_ref = 1.0
    chirp_mass = parameters["chirp_mass"]
    mass_factor = (chirp_mass / m_ref) ** (5 / 6)

    if jacobian:
        return -torch.log(mass_factor)

    if reverse:
        parameters["distance"] = parameters["distance"] / mass_factor
    else:
        parameters["distance"] = parameters["distance"] * mass_factor

    return parameters
