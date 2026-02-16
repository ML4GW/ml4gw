import math

import torch
from torch import Tensor

from ml4gw.types import BatchTensor


def tukey_window(
    n: int, alpha: float = 0.5, device=None, dtype=None
) -> torch.Tensor:
    """
    Generate a length-n Tukey window with
    fraction alpha of the window tapered.
    alpha=0.5 => 25% of the samples on each end are tapered,
    50% are flat in the middle.
    """
    w = torch.ones(n, device=device, dtype=dtype)
    if alpha <= 0:
        return w  # no taper
    if alpha >= 1:
        # Entire window is a Hann window
        t = torch.linspace(0, math.pi, n, device=device, dtype=dtype)
        return 0.5 * (1.0 - torch.cos(t))

    # Taper fraction
    taper_len = int(alpha * (n - 1) / 2.0)
    # first taper
    t = torch.linspace(0, math.pi / 2, taper_len, device=device, dtype=dtype)
    w[:taper_len] = 0.5 * (1 - torch.cos(t))
    # last taper
    t = torch.linspace(math.pi / 2, 0, taper_len, device=device, dtype=dtype)
    w[-taper_len:] = 0.5 * (1 - torch.cos(t))
    return w


class GenerateString(torch.nn.Module):
    """
    PyTorch re-implementation of the 'XLALGenerateString'
    logic for cosmic-string waveforms: 'cusp', 'kink', or 'kinkkink'.
    LAL sets:
      - f_low = 1 Hz
      - length = floor(9.0 / f_low / dt / 2) * 2 + 1  => ~9 seconds total
      - The waveforms are built in frequency domain and iFFT to time domain.
    """

    def __init__(self, sample_rate: float, duration: float, device="cpu"):
        """
        Args:
            sample_rate: sampling rate (Hz).
            device: which device ("cpu" or "cuda") to store buffers on.
        """
        super().__init__()
        self.sample_rate = sample_rate

        dt = 1.0 / sample_rate
        f_low = 1.0  # Hard-coded in the LAL code

        # We want length = 2 * floor( (9.0/f_low)/dt/2 ) + 1,
        # doing the arithmetic as GPU-friendly if desired.
        # But to create an actual Python shape,
        # we do .cpu().item() at the end.
        sim_duration = 9.0
        if duration > 4.5:
            sim_duration = duration * 2
        temp = (sim_duration / f_low) / dt / 2.0  # float in Python
        # If you want GPU math, push it to a GPU tensor first:
        temp_t = torch.tensor(temp, device=device, dtype=torch.float32)
        half_len = torch.floor(temp_t)  # GPU-based floor
        half_len_int = int(half_len.cpu().item())  # bring back to Python int
        length = 2 * half_len_int + 1

        self.length = length
        self.dt = dt
        self.f_low = f_low

        # Frequency bins for real-FFT => 0 .. length//2
        freq_bins = length // 2 + 1
        df = 1.0 / (length * dt)

        # Build freq array and the phase factor
        freq = torch.arange(freq_bins, dtype=torch.float64, device=device) * df
        k = torch.arange(freq_bins, dtype=torch.float64, device=device)
        phase_factor = torch.exp(
            -1j * math.pi * k * (length - 1) / float(length)
        )

        # Calculate time window length
        win_len = int(sample_rate * duration)
        half_win_len = int(win_len / 2)

        self.win_start = half_len_int - half_win_len
        self.win_end = half_len_int + half_win_len

        # Register these as buffers so they move with model.to(device)
        self.register_buffer("freq", freq)  # shape (freq_bins,)
        self.register_buffer(
            "phase_factor", phase_factor
        )  # shape (freq_bins,)

        # Build final Tukey(0.5) window for time domain
        tw = tukey_window(
            length, alpha=0.5, device=device, dtype=torch.float64
        )
        self.register_buffer("tukey", tw)  # shape (length,)

    def forward(
        self, power: float, amplitude: BatchTensor, f_high: float
    ) -> (Tensor, Tensor):
        """
        Generate the chosen cosmic-string waveform in plus polarization,
        with cross=0.
        waveform must be:
            "cusp"  -4.0 / 3.0,
            "kink" -5.0 / 3.0, or
            "kinkkink" -2.0.
        Args:
            power: cusp = -4.0 / 3.0, kink = -5.0 / 3.0, or kinkkink = -2.0.
            amplitude: (batch,) overall amplitude scaling parameter.
            f_high: (batch,) freq above which we apply exponential taper.
        Returns:
            (h_cross, h_plus): shape (batch, self.length).
            The cross polarization is zero (as in LAL).
        """
        # Reshape for batch dimension
        amplitude = amplitude.view(-1, 1)

        device = self.freq.device
        # We'll assume amplitude is already on the same device
        # or you can do amplitude = amplitude.to(device)

        length = self.length
        freq_bins = self.freq.shape[0]  # length//2 + 1

        # Expand freq => (1, freq_bins), phase_factor => (1, freq_bins)
        freq = self.freq.unsqueeze(0)  # shape (1, freq_bins)
        phase_factor = self.phase_factor.unsqueeze(0)  # shape (1, freq_bins)

        # DC and Nyquist bins are zero => fill only 1..freq_bins-2
        valid_mask = torch.ones(freq_bins, dtype=torch.bool, device=device)
        valid_mask[0] = False
        valid_mask[-1] = False
        valid_mask_2d = valid_mask.unsqueeze(0)  # (1, freq_bins)

        # Avoid divide-by-zero
        f_clamped = torch.clamp(freq, min=1e-20)

        # Common factor => (1 + (f_low^2)/(f^2))^-4
        base_factor = (1.0 + (self.f_low**2) / (f_clamped**2)) ** (-4.0)

        # Multiply by f^power
        base_factor *= f_clamped ** (power)

        # Taper above f_high => if freq>f_high => exp(1 - freq/f_high), else 1
        ratio = freq / f_high  # (batch, freq_bins)
        taper = torch.where(
            ratio > 1.0, torch.exp(1.0 - ratio), torch.ones_like(ratio)
        )

        # Combine =>
        # amplitude * base_factor * taper => shape (batch, freq_bins)
        amp_val = amplitude * base_factor
        amp_val = amp_val * taper

        # Zero out DC/Nyquist
        amp_val = torch.where(
            valid_mask_2d, amp_val, torch.zeros_like(amp_val)
        )

        # Multiply by phase factor
        A = amp_val * phase_factor  # shape (batch, freq_bins), complex128

        # iFFT => time domain: shape (batch, length), real
        hplus = torch.fft.irfft(
            A, n=length, dim=-1
        )  # THIS SHOULD BE NORM='FORWARD' BUT IT DOESNT GIVE THE SAME AS LAL
        hplus = (
            hplus * self.sample_rate
        )  # scale by dt THIS IS WEIRD ONLY THIS WORKS
        hcross = torch.zeros_like(hplus)

        # Apply Tukey(0.5) window to plus
        w = self.tukey.unsqueeze(0).to(dtype=hplus.dtype)
        hplus = hplus * w

        hcross = hcross[..., self.win_start : self.win_end]
        hplus = hplus[..., self.win_start : self.win_end]
        return (hcross, hplus)