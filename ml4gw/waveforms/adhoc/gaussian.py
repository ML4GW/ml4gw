import torch
from torch import Tensor
from ml4gw.types import BatchTensor

class Gaussian(torch.nn.Module):
    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration

        num_samples = int(sample_rate * duration)
        times = torch.arange(num_samples, dtype=torch.float64) / sample_rate
        times -= 0.5 * duration

        self.register_buffer("times", times)

    def forward(
        self,
        hrss: Tensor,
        polarization: Tensor,
        eccentricity: Tensor,
    ):

        hrss = hrss.view(-1, 1)
        polarization = polarization.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)  # not used

        # Basic definitions
        hplusrss  = hrss * torch.cos(polarization)
        hcrossrss = hrss * torch.sin(polarization)

        # If we want "2 / sqrt(pi)" via PyTorch, do:
        # make pi a tensor, matching the device/dtype of 'hrss'
        pi_t = torch.tensor(torch.pi, dtype=hrss.dtype, device=hrss.device)
        FRTH_2_Pi = 2.0 / torch.sqrt(pi_t)

        # Also, we want sqrt(self.duration) as a tensor
        dur_t = torch.tensor(self.duration, dtype=hrss.dtype, device=hrss.device)
        sigma = torch.sqrt(dur_t)

        # Then compute the "peak" amplitudes
        h0_plus  = hplusrss  / sigma * FRTH_2_Pi
        h0_cross = hcrossrss / sigma * FRTH_2_Pi

        # Evaluate Gaussian factor
        # 'self.times' might be on CPU if you didn't move the model to GPU;
        # to be safe, ensure it matches 'hrss' device/dtype:
        t = self.times.to(hrss.device, hrss.dtype)  # shape (num_samples,)
        t2 = t**2
        t2 = t2.unsqueeze(0)  # shape (1, num_samples)

        # again we can do a broadcast with float 'self.duration' or
        # we can just re-use 'dur_t':
        fac = torch.exp(-t2 / (dur_t**2))

        plus = h0_plus * fac
        cross = h0_cross * fac

        return cross, plus