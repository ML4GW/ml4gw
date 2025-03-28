import torch
from torch import Tensor
from ml4gw.types import BatchTensor


def semi_major_minor_from_e(e: float) -> (Tensor, Tensor):
    """
    e -> (a, b), used in the elliptical amplitude factor.
    a = sqrt(1 + e),  b = sqrt(1 - e).
    We'll do the "sqrt()" in PyTorch, so pass 'e' as a PyTorch float or
    wrap in a function that returns Tensors.
    """
    # If 'e' is a Python float, we can do
    #   e_t = torch.tensor(e, dtype=...)
    # But if we just want the functional form, do:
    a = torch.sqrt(1.0 + e)
    b = torch.sqrt(1.0 - e)
    return a, b


class WhiteNoiseBurst(torch.nn.Module):
    """
    PyTorch re-implementation of XLALGenerateBandAndTimeLimitedWhiteNoiseBurst.

    On every forward() call, draws new noise in time domain, applies
    Gaussian windows (time & frequency), normalizes derivative-power,
    and returns time-domain h_cross(t), h_plus(t).

    All numerical steps parallel the LAL code logic:
      - Time array length = 21 * duration, up to rounding => floor(...)*2+1
      - Window in time with effective sigma = sqrt(duration^2/4 - 1/(pi^2 * bandwidth^2)).
      - Gaussian window in freq domain around 'frequency' with width=bandwidth/2.
      - Phase rotation and elliptical amplitude from eccentricity, phase.
      - Normalize so ∫ (ḣ₊² + ḣ×²) dt = int_hdot_squared.
      - Final Tukey(0.5) window for continuity at edges.
    """

    def __init__(self, sample_rate: float, duration: float, device="cpu"):
        """
        Args:
            sample_rate: The sampling rate (Hz).
            duration:    The nominal 'duration' of the burst (seconds).
            device:      "cpu" or "cuda". This sets where we store buffers.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.duration = duration  # Keep as a float attribute
        self.dt = 1.0 / sample_rate

        # LAL logic: length = floor(21.0 * duration / dt / 2.0)*2 + 1
        # We'll do it in a GPU-friendly way:
        temp = torch.tensor(21.0 * self.duration / self.dt / 2.0, device=device, dtype=torch.float32)
        half_len_floor = torch.floor(temp)          # GPU-based floor
        half_len_int = int(half_len_floor.cpu().item())   # bring scalar to CPU
        length = 2 * half_len_int + 1

        self.length = length  # store as a Python int

        # Build a time array of shape (length,) on the chosen device
        # from - (length-1)/2 to + (length-1)/2
        times = torch.arange(length, device=device, dtype=torch.float64)
        times = times * self.dt
        times -= 0.5 * self.dt * (length - 1)

        # Register as a buffer so that it moves automatically with .to(...)
        self.register_buffer("times", times)

    def forward(
        self,
        frequency: BatchTensor,
        bandwidth: BatchTensor,
        eccentricity: BatchTensor,
        phase: BatchTensor,
        int_hdot_squared: BatchTensor
    ):
        """
        Generate a band/time-limited White Noise Burst.

        Args:
            frequency: (batch,) Center frequency (Hz).
            bandwidth: (batch,) 1-sigma freq extent => (bandwidth/2) is actual Gaussian width in freq domain.
            eccentricity: (batch,) e in [0, 1], sets elliptical amplitude factors.
            phase: (batch,) Overall phase offset (radians).
            int_hdot_squared: (batch,) Desired ∫(ḣ₊² + ḣ×²) dt.

        Returns:
            (h_cross, h_plus) of shape (batch, length).
        """
        # 1) Reshape inputs => (batch, 1)
        frequency = frequency.view(-1, 1)
        bandwidth = bandwidth.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)
        phase = phase.view(-1, 1)
        int_hdot_squared = int_hdot_squared.view(-1, 1)

        batch = frequency.shape[0]
        length = self.length
        dt = self.dt
        device = frequency.device
        dtype = frequency.dtype

        # 2) Draw random time-domain noise => (batch, length)
        #    Each pol is an independent N(0,1).
        hplus_time = torch.randn(batch, length, device=device, dtype=dtype)
        hcross_time = torch.randn(batch, length, device=device, dtype=dtype)

        # 3) Time-domain Gaussian window
        #    sigma_t² = duration²/4 - 1/(π² * bandwidth²).
        #    clamp to min=1e-20 to avoid negative.
        sigma_t_sq = (self.duration**2 / 4.0) - 1.0 / (torch.pi**2 * bandwidth.squeeze(-1) ** 2)
        sigma_t_sq = torch.clamp(sigma_t_sq, min=1e-20)
        sigma_t = sigma_t_sq.sqrt().unsqueeze(-1)  # shape (batch, 1)

        # times => shape (length,) => expand => (1,length)
        t = self.times.unsqueeze(0).to(dtype=dtype)  # shape (1, length)
        # Multiply by e^(- t² / sigma_t²)
        w_time = torch.exp(- (t**2) / (sigma_t**2))

        hplus_time *= w_time
        hcross_time *= w_time

        # 4) Forward rFFT => shape (batch, length//2 + 1)
        Hplus_freq = torch.fft.rfft(hplus_time, dim=-1)
        Hcross_freq = torch.fft.rfft(hcross_time, dim=-1)

        # 5) Frequency-domain Gaussian window around "frequency", with sigma_f = bandwidth/2
        #    plus elliptical amplitude & phase
        freq_bins = Hplus_freq.shape[-1]
        df = self.sample_rate / length
        k = torch.arange(freq_bins, device=device, dtype=dtype).unsqueeze(0)  # shape (1, freq_bins)
        f_array = k * df - frequency  # (batch, freq_bins)

        beta = -0.5 / ((bandwidth / 2.0) ** 2)
        w_freq = torch.exp(f_array**2 * beta)  # (batch, freq_bins)

        # eccentricity => a=√(1+e), b=√(1−e)
        # We'll do it per batch.  If e is shape (batch,1), we can do:
        a = torch.sqrt(1.0 + eccentricity)
        b = torch.sqrt(1.0 - eccentricity)

        freq_bins = Hplus_freq.shape[-1]
        batch = Hplus_freq.shape[0]

        k = torch.arange(freq_bins, device=device, dtype=dtype)           # shape (freq_bins,)

        # Make a DC mask => shape (1, freq_bins) => (batch, freq_bins)
        dc_mask = (k == 0).unsqueeze(0).expand(batch, freq_bins)  # 2D, bool

        # Build rot_phase_2d => shape (batch, freq_bins)
        rot_phase_2d = torch.exp(-1j * phase).expand(-1, freq_bins)

        # We'll define '1' as a complex scalar in the same dtype:
        one_cplx = torch.ones_like(rot_phase_2d)  # shape (batch, freq_bins), complex

        # For plus => factor=1 at DC bin, otherwise exp(-i phase)
        rot_factor_plus = torch.where(dc_mask, one_cplx, rot_phase_2d)

        # For cross => factor=1 at DC bin, otherwise i * exp(-i phase)
        i_factor = torch.tensor(1.0j, device=device, dtype=rot_phase_2d.dtype)
        rot_factor_cross = i_factor * rot_phase_2d
        rot_factor_cross = torch.where(dc_mask, one_cplx, rot_factor_cross)

        # Multiply
        Hplus_freq *= rot_factor_plus
        Hcross_freq *= rot_factor_cross

        # 6) Compute current ∫(ḣ²) dt => measure derivative power
        # derivative => (i 2π f_phys) in freq domain
        def int_hdot_sq(Hf: torch.Tensor) -> torch.Tensor:
            # Hf shape(batch, freq_bins), freq => k*df
            f_phys = k  * df  # shape(1, freq_bins)
            factor = (2.0 * torch.pi * f_phys)**2  # shape(1, freq_bins)
            amp2 = torch.abs(Hf)**2  # shape(batch, freq_bins)
            return torch.sum(factor * amp2, dim=-1) * df  # shape(batch,)

        curr_hdotsq = int_hdot_sq(Hplus_freq) + int_hdot_sq(Hcross_freq)
        desired = int_hdot_squared.squeeze(-1)  # shape(batch,)

        eps = 1e-20
        norm_factor = torch.sqrt(curr_hdotsq / (desired + eps))
        norm_factor = torch.clamp(norm_factor, min=eps)
        norm_factor = norm_factor.unsqueeze(-1)  # shape(batch,1)

        Hplus_freq /= norm_factor
        Hcross_freq /= norm_factor

        # 7) iFFT => time domain
        hplus_time = torch.fft.irfft(Hplus_freq, n=length, dim=-1)
        hcross_time = torch.fft.irfft(Hcross_freq, n=length, dim=-1)

        # 8) Final Tukey(0.5) window
        def tukey_window(n, alpha=0.5, device=None, dtype=None):
            w = torch.ones(n, device=device, dtype=dtype)
            if alpha <= 0:
                return w
            if alpha >= 1:
                t2 = torch.linspace(0, torch.pi, n, device=device, dtype=dtype)
                return 0.5*(1.0 - torch.cos(t2))

            taper_len = int(alpha*(n-1)/2)
            t1 = torch.linspace(0, torch.pi/2, taper_len, device=device, dtype=dtype)
            w[:taper_len] = 0.5*(1.0 - torch.cos(t1))
            t2 = torch.linspace(torch.pi/2, 0, taper_len, device=device, dtype=dtype)
            w[-taper_len:] = 0.5*(1.0 - torch.cos(t2))
            return w

        tw = tukey_window(length, alpha=0.5, device=device, dtype=dtype).unsqueeze(0)
        hplus_time *= tw
        hcross_time *= tw

        # Return (h_cross, h_plus)
        return hcross_time, hplus_time