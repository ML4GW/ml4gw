import torch

from ml4gw.types import ScalarTensor, TimeSeriesTensor


class Pols_from_SDQM_3d(torch.nn.Module):

    # Reference:
    # https://academic.oup.com/ptps/article/doi/10.1143/PTPS.128.183/1930275

    """
    Callable class for generating cross and plus polarizations from
    3D simulated second-derivative quadrupole moment.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self):

        super().__init__()

    def forward(
        self,
        sqdm: TimeSeriesTensor,
        theta: ScalarTensor,
        phi: ScalarTensor,
    ):
        """
        Generate polarizations waveform based on the orientation.
        The sphere cordinated follows physics convention.
        See the following link for more detail.

        link = ('https://en.wikipedia.org/wiki/Spherical_coordinate_system#
        /media/File:3D_Spherical.svg')

        Args:
            sqdm:
                Second-derivative quadrupole moment.
            theta:
                Polar angle. Range [0, pi]
            phi:
                Azimuthal angle. Range [0, 2 * pi)
        Returns:
            Tensors of cross and plus polarizations
        """

        ori_matrix_c = torch.zeros([len(theta), 3, 3])
        ori_matrix_p = torch.zeros([len(theta), 3, 3])

        # ori_matrix_c
        ori_matrix_c[:, 0, 0] = -2 * (
            torch.cos(theta) * torch.sin(phi) * torch.cos(phi)
        )
        ori_matrix_c[:, 0, 1] = 2 * torch.cos(theta) * torch.cos(2 * phi)
        ori_matrix_c[:, 1, 1] = (
            2 * torch.cos(theta) * torch.sin(phi) * torch.cos(phi)
        )
        ori_matrix_c[:, 1, 2] = -2 * (torch.sin(theta) * torch.cos(phi))
        ori_matrix_c[:, 2, 0] = 2 * torch.sin(theta) * torch.sin(phi)

        # ori_matrix_p
        ori_matrix_p[:, 0, 0] = (
            torch.cos(theta) ** 2 * torch.cos(phi) ** 2 - torch.sin(phi) ** 2
        )
        ori_matrix_p[:, 0, 1] = torch.cos(theta) ** 2 * torch.sin(
            2 * phi
        ) - torch.sin(2 * phi)
        ori_matrix_p[:, 1, 1] = (
            torch.cos(theta) ** 2 * torch.sin(phi) ** 2 - torch.cos(phi) ** 2
        )
        ori_matrix_p[:, 1, 2] = -(torch.sin(2 * theta) * torch.sin(phi))
        ori_matrix_p[:, 2, 0] = -(torch.sin(2 * theta) * torch.cos(phi))
        ori_matrix_p[:, 2, 2] = torch.sin(theta) ** 2

        h_cross = torch.einsum("kji,nij->nk", sqdm, ori_matrix_c)
        h_plus = torch.einsum("kji,nij->nk", sqdm, ori_matrix_p)

        return h_cross, h_plus
