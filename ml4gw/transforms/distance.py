import torch


def sample_chirp_distance(parameters, reverse=False, jacobian=False):
    """
    Convert the uniform chirp distance (Dc) to luminosity distance (Dl)
    Use only chirp mass scaling
    """
    # mass factor M_f = (Mc/M_ref)^(5/6)
    M_ref = 1.0
    chirp_mass = parameters["chirp_mass"]
    M_f = (chirp_mass / M_ref) ** (5 / 6)

    if not reverse and not jacobian:
        chirp_distance = parameters["distance"]  # Dc (uniform from priors)
        # Dl = Dc * M_f
        parameters["distance"] = chirp_distance * M_f

    if reverse:
        distance = parameters["distance"]  # Dl (mass scaled)
        # Dc = Dl / M_f
        parameters["distance"] = distance / M_f

    if jacobian:
        return -torch.log(M_f)
    else:
        return parameters
