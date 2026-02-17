import torch

from ml4gw.types import BatchTensor


class Gaussian(torch.nn.Module):
    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration

        # 2 Δt samples, centred on zero
        k_len: float = 2.0
        num_samples = int(round(k_len * duration * sample_rate)) + 1
        half_num_samples = int(num_samples / 2)
        t = (
            torch.arange(num_samples, dtype=torch.float64)
            - (num_samples - 1) / 2
        ) / sample_rate
        self.register_buffer("times", t)

        # Calculate time window length
        win_len = int(sample_rate * duration)
        half_win_len = int(win_len / 2)

        self.win_start = half_num_samples - half_win_len
        self.win_end = half_num_samples + half_win_len

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

    def forward(
        self,
        hrss: BatchTensor,
        gaussian_width: BatchTensor,
    ):
        # The gaussian_width should be smaller than self.duration
        hrss = hrss.view(-1, 1)
        gaussian_width = gaussian_width.view(-1, 1)

        # correct LAL normalisation
        h0 = hrss / torch.sqrt(
            torch.sqrt(
                torch.tensor(torch.pi, dtype=hrss.dtype, device=hrss.device)
            )
            * gaussian_width
        )

        h0_plus = h0
        h0_cross = h0 * 0

        t = self.times.to(hrss.device, hrss.dtype)
        env = torch.exp(-0.5 * t.pow(2).view(1, -1) / gaussian_width**2)

        win = self.tukey.to(hrss.device, hrss.dtype)[None, :]

        plus = (h0_plus * env) * win
        cross = (h0_cross * env) * win

        cross = cross[..., self.win_start : self.win_end]
        plus = plus[..., self.win_start : self.win_end]
        return cross, plus
