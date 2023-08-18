import torch


# based on values from
# https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html
class InterferometerGeometry:
    def __init__(self, name: str):
        if name == "H1":
            self.x_arm = torch.Tensor(
                (-0.22389266154, +0.79983062746, +0.55690487831)
            )
            self.y_arm = torch.Tensor(
                (-0.91397818574, +0.02609403989, -0.40492342125)
            )
            self.vertex = torch.Tensor(
                (-2.16141492636e06, -3.83469517889e06, +4.60035022664e06)
            )
        elif name == "L1":
            self.x_arm = torch.Tensor(
                (-0.95457412153, -0.14158077340, -0.26218911324)
            )
            self.y_arm = torch.Tensor(
                (+0.29774156894, -0.48791033647, -0.82054461286)
            )
            self.vertex = torch.Tensor(
                (-7.42760447238e04, -5.49628371971e06, +3.22425701744e06)
            )
        elif name == "V1":
            self.x_arm = torch.Tensor(
                (-0.70045821479, +0.20848948619, +0.68256166277)
            )
            self.y_arm = torch.Tensor(
                (-0.05379255368, -0.96908180549, +0.24080451708)
            )
            self.vertex = torch.Tensor(
                (4.54637409900e06, 8.42989697626e05, 4.37857696241e06)
            )
        else:
            raise ValueError(
                f"{name} is not recognized as an interferometer, "
                "or has not been implemented"
            )
