import math

import torch
from torch import nn

from ml4gw.types import BatchTensor  # typically an alias for torch.Tensor

torch.manual_seed(42)


def semi_major_minor_from_e(eccentricity):
    e2 = eccentricity * eccentricity
    a = 1.0 / torch.sqrt(2.0 - e2)
    b = a * torch.sqrt(1.0 - e2)
    return a, b


def tukey_window(n, alpha=0.5, device="cpu", dtype=torch.float64):
    w = torch.ones(n, device=device, dtype=dtype)
    if alpha <= 0:
        return w
    if alpha >= 1:
        t_lin = torch.linspace(0, torch.pi, n, device=device, dtype=dtype)
        return 0.5 * (1.0 - torch.cos(t_lin))
    taper_len = int(alpha * (n - 1) / 2)
    t1 = torch.linspace(0, torch.pi / 2, taper_len, device=device, dtype=dtype)
    w[:taper_len] = 0.5 * (1.0 - torch.cos(t1))
    t2 = torch.linspace(torch.pi / 2, 0, taper_len, device=device, dtype=dtype)
    w[-taper_len:] = 0.5 * (1.0 - torch.cos(t2))
    return w


class WhiteNoiseBurst(nn.Module):
    """
    Faithful PyTorch re-implementation of
    XLALGenerateBandAndTimeLimitedWhiteNoiseBurst.

    On each forward() call, new independent white-noise bursts are generated
    (one for h₊ and one for hₓ) following these steps:
      - Compute the time-series length:
        floor(k_len * duration / delta_t / 2)*2 + 1.
        lal's implementation uses k_len=21,
        but we change it to 7 to reduce the computational workload.
      - Apply a time-domain Gaussian window with:
        effective sigma = sqrt(duration²/4 - 1/(π² * bandwidth²)).
      - Transform to the frequency domain (rFFT).
      - Apply a frequency-domain Gaussian envelope centered at 'frequency' with
        width = bandwidth/2), and adjust amplitudes with
        elliptical factors:
            a = √(2 - eccentricity^2) for h₊,
            b = √(1 – eccentricity^2) for hₓ.
      - For non-DC bins,
      rotate the phase by exp( – i·phase) for h₊ and by i·exp(–i·phase) for hₓ.
      - Normalize so that ∫(ḣ₊²+ḣₓ²)dt equals int_hdot_squared.
      - Inverse FFT back to the time domain
        and apply a final Tukey window (α=0.5) to smooth the edges.
    """

    def __init__(self, sample_rate: float, duration: float):
        """
        Args:
            sample_rate: Sampling rate in Hz.
            duration: Nominal burst duration in seconds.
        """
        super().__init__()
        k_len: float = 7.0
        self.sample_rate = sample_rate
        self.duration = duration
        self.dt = 1.0 / sample_rate

        # Determine time-series length
        # length = floor(21.0*duration/dt/2)*2 + 1.
        half_len = math.floor(k_len * duration / self.dt / 2.0)
        self.length = 2 * half_len + 1

        win_len = int(sample_rate * duration)
        half_win_len = int(win_len / 2)

        self.win_start = half_len - half_win_len
        self.win_end = half_len + half_win_len

        # Build a time axis, centered so that the middle sample is t = 0.
        times = torch.arange(self.length, dtype=torch.float64) * self.dt
        times = times - 0.5 * self.dt * (self.length - 1)
        self.register_buffer("times", times)

    def forward(
        self,
        time_envelope: BatchTensor,
        frequency: BatchTensor,
        bandwidth: BatchTensor,
        eccentricity: BatchTensor,
        phase: BatchTensor,
        int_hdot_squared: BatchTensor,
    ):
        """
        Generate a band- and time-limited white noise burst.

        Args:
            frequency: (batch,) Center frequency (Hz).
            bandwidth: (batch,) Frequency-domain 1-σ extent (Hz);
            Gaussian envelope has width = bandwidth/2.
            eccentricity: (batch,) Value in [0, 1].
            phase: (batch,) Overall phase offset (radians).
            int_hdot_squared: (batch,) Desired ∫(ḣ₊² + ḣₓ²) dt.

        Returns:
            A tuple (h_cross, h_plus), each of shape (batch, length).
        """
        # Reshape all inputs to (batch, 1)
        time_envelope = time_envelope.view(-1, 1)
        frequency = frequency.view(-1, 1)
        bandwidth = bandwidth.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)
        phase = phase.view(-1, 1)
        int_hdot_squared = int_hdot_squared.view(-1, 1)

        batch = frequency.shape[0]
        length = self.length
        device = frequency.device
        dtype = frequency.dtype

        # --- Input validation ---
        if (
            (self.duration < 0)
            or (frequency < 0).any()
            or (bandwidth < 0).any()
            or (eccentricity < 0).any()
            or (eccentricity > 1).any()
            or (int_hdot_squared < 0).any()
        ):
            raise ValueError("Invalid input parameters.")

        # New with batched duration
        sigma_t_sq = (time_envelope.squeeze(-1) ** 2 / 4.0) - 1.0 / (
            torch.pi**2 * (bandwidth.squeeze(-1) ** 2)
        )
        if (sigma_t_sq < 0).any():
            raise ValueError(
                "Invalid input parameters: sigma_t² < 0 "
                "(time_envelope * bandwidth too small)."
            )
        # Effective time-domain sigma in seconds per batch sample.
        sigma_t = sigma_t_sq.sqrt()  # shape (batch,)

        # --- Generate time-domain noise and apply time Gaussian window ---
        hplus = torch.randn(batch, length, device=device, dtype=dtype)
        hcross = torch.randn(batch, length, device=device, dtype=dtype)
        # Use the precomputed time axis (centered at 0); shape: (1, length)
        t_row = self.times.to(dtype=dtype).unsqueeze(0)
        # Gaussian window:
        # w(t) = exp(-0.5*(t/σ)²) with σ from sigma_t (per batch).
        w_time = torch.exp(-0.5 * (t_row / sigma_t.unsqueeze(1)) ** 2)
        hplus = hplus * w_time
        hcross = hcross * w_time
        # --- Forward FFT (rFFT) ---
        Hplus = torch.fft.rfft(hplus, dim=-1)
        Hcross = torch.fft.rfft(hcross, dim=-1)
        # --- Frequency-domain processing ---
        nfreq = Hplus.shape[-1]
        df = self.sample_rate / length
        # Frequency axis:
        # f = k*df, where k = 0,..., nfreq-1. Shape: (1, nfreq)
        k = torch.arange(nfreq, device=device, dtype=dtype).unsqueeze(0)
        f_array = k * df
        # Shift frequency so that the envelope is centered at 'frequency'
        f_offset = f_array - frequency  # broadcasts to (batch, nfreq)
        # Gaussian envelope parameter: beta = -0.5 / ((bandwidth/2)²)
        beta = -0.5 / (
            (bandwidth / 2.0) ** 2
        )  # shape (batch, 1) by broadcasting
        w_freq = torch.exp((f_offset**2) * beta)  # (batch, nfreq)

        a, b = semi_major_minor_from_e(eccentricity)

        # Multiply frequency-domain series by
        # the envelope and elliptical factors.
        Hplus = Hplus * (a * w_freq)
        Hcross = Hcross * (b * w_freq)

        # For non-DC bins, rotate phase:
        # for h₊ multiply by exp(-i·phase), for hₓ by i·exp(-i·phase).
        # Create a (batch, nfreq) mask that is True for non-DC bins.
        k_full = (
            torch.arange(nfreq, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch, nfreq)
        )
        non_dc = k_full != 0
        # Compute phase factor (batch, 1) then broadcast.
        pf = torch.exp(-1j * phase)
        Hplus = torch.where(non_dc, Hplus * pf, Hplus)
        Hcross = torch.where(non_dc, Hcross * (1j * pf), Hcross)

        # --- Normalization ---
        # In frequency domain, differentiation multiplies by (i 2π f);
        # so squared magnitude is scaled by (2π f)².
        # f_phys: shape (1, nfreq)
        f_phys = k * df
        factor = (2 * torch.pi * f_phys) ** 2
        factor *= 2  # Account for negitive frequency
        power_plus = torch.sum(factor * (torch.abs(Hplus) ** 2), dim=-1) * df
        power_cross = torch.sum(factor * (torch.abs(Hcross) ** 2), dim=-1) * df
        current_hdotsq = power_plus + power_cross  # shape (batch,)
        eps = 1e-50

        norm_factor = torch.sqrt(
            current_hdotsq / (int_hdot_squared.squeeze(-1) + eps)
        )
        norm_factor = norm_factor.clamp(min=eps).unsqueeze(-1)
        Hplus = Hplus / norm_factor
        Hcross = Hcross / norm_factor

        # --- Inverse FFT to get time-domain burst ---
        hplus_time = (
            torch.fft.irfft(Hplus, n=length, dim=-1) * self.sample_rate
        )
        hcross_time = (
            torch.fft.irfft(Hcross, n=length, dim=-1) * self.sample_rate
        )

        tw = tukey_window(
            length, alpha=0.5, device=device, dtype=dtype
        ).unsqueeze(0)
        hplus_time = hplus_time * tw
        hcross_time = hcross_time * tw

        hcross_time = hcross_time[..., self.win_start : self.win_end]
        hplus_time = hplus_time[..., self.win_start : self.win_end]
        return hcross_time, hplus_time