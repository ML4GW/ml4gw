import torch
from torch import Tensor
from ml4gw.types import BatchTensor

class Gaussian(torch.nn.Module):
    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration                      # = Δt in the LAL docs

        # 21 Δt samples, centred on zero
        num_samples = int(round(21 * duration * sample_rate))
        t = (torch.arange(num_samples, dtype=torch.float64) -
             (num_samples - 1) / 2) / sample_rate
        self.register_buffer("times", t)

        # Tukey window with α = 0.5
        alpha = 0.5
        n = num_samples
        tukey = torch.ones(n, dtype=torch.float64)
        k = int(alpha * (n - 1) / 2)
        if k > 0:
            tau = torch.linspace(0, torch.pi, k + 1)[:-1]
            tukey[:k] = 0.5 * (1 - torch.cos(tau))
            tukey[-k:] = tukey[:k].flip(0)
        self.register_buffer("tukey", tukey)

    def forward(self, hrss: Tensor, polarization: Tensor, eccentricity: Tensor, duration: Tensor):
        hrss       = hrss.view(-1, 1)
        psi        = polarization.view(-1, 1)
        duration = duration.view(-1, 1)

        # correct LAL normalisation
        h0 = hrss / torch.sqrt(torch.sqrt(torch.tensor(torch.pi,
                                   dtype=hrss.dtype, device=hrss.device))
                               * self.duration)

        h0_plus  = h0 * torch.cos(psi)
        h0_cross = h0 * torch.sin(psi)

        t = self.times.to(hrss.device, hrss.dtype)
        #env = torch.exp(-0.5 * t.pow(2) / self.duration**2)[None, :]
        env = torch.exp(-0.5 * t.pow(2).view(1,-1) / duration**2)

        win = self.tukey.to(hrss.device, hrss.dtype)[None, :]

        #print("Gaussian \t env",env.shape, "win", win.shape, "h0_plus", h0_plus.shape, "h0_cross", h0_cross.shape)
        plus  = (h0_plus  * env) * win
        cross = (h0_cross * env) * win
        return cross, plus