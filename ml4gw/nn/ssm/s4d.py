# Adapted with modifications from https://github.com/state-spaces/s4
# (models/s4/s4d.py), licensed under Apache-2.0.
# Detailed description of the S4D model and its parameters:
# "On the Parameterization and Initialization of Diagonal State Space Models"
# (https://arxiv.org/abs/2206.11893)

import math

import torch
import torch.nn as nn


class S4DKernel(nn.Module):
    """Build the convolution kernel for one S4D layer.

    Each channel is an independent diagonal linear time-invariant
    state-space model, defined in continuous time by

        x'(t) = A * x(t) + B * u(t)
        y(t)  = C * x(t)

    with A diagonal. Linear time invariance means the output equals the
    input convolved with a single kernel K, so the whole sequence is
    processed in one convolution instead of stepped through position by
    position. Discretizing with timestep dt gives the kernel this module
    returns:

        K_l = 2 * Re(
            sum_n C_n * ((exp(dt * A_n) - 1) / A_n) * exp(dt * A_n)^l
        ),
        l = 0, ..., L-1   (n runs over the N/2 states)

    Args:
        d_model: Model dimension.
            Equivalent to the number of channels in CNN nomenclature,
            i.e. independent SSMs created.
        N: State size per model dimension.
            The states are stored as N / 2 conjugate pairs, so N must be even.
        dt_min: Lower bound of the per-channel timestep, sampled
            log-uniformly in [dt_min, dt_max]. The timestep sets the
            timescale each SSM resolves.
        dt_max: Upper bound of the per-channel timestep.
        lr: Optional learning rate for `dt` and `A` parameters.
            If None, these parameters are optimized with the same LR as the
            rest of the model. If 0.0, these parameters are frozen.
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()

        H = d_model  # Number of independent SSMs (one per channel).

        ### SSM parameter initialization ###
        # For each channel, draw a random timestep `dt`
        # from a log-uniform distribution.
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        # Initialize the SSM output matrix `C` with i.i.d. Gaussian entries.
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        # Initialize the SSM state matrix `A` with S4D-Lin initialization
        # (Re(A) = -0.5, Im(A) = pi * n).
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * torch.arange(N // 2).repeat(H, 1)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L: int) -> torch.Tensor:
        """Returns: (H, L) convolution kernel."""

        dt = torch.exp(self.log_dt)  # (H,)
        C = torch.view_as_complex(self.C)  # (H, N//2)
        # torch.complex(...) instead of `1j` for torch.compile safety
        # Re(A) = -exp(log_A_real) is always < 0, so modes decay (stable)
        A = torch.complex(
            -torch.exp(self.log_A_real), self.A_imag
        )  # (H, N//2)

        dtA = A * dt.unsqueeze(-1)  # (H, N//2)
        K = dtA.unsqueeze(-1) * torch.arange(
            L, device=A.device
        )  # (H, N//2, L)

        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real  # (H, L)
        return K

    def register(
        self, name: str, tensor: torch.Tensor, lr: float | None = None
    ) -> None:
        """Register a tensor with a configurable LR and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim


class S4D(nn.Module):
    """Single S4D layer operating on (B, H, L) sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(d_model))

        self.kernel = S4DKernel(
            d_model, N=d_state, dt_min=dt_min, dt_max=dt_max, lr=lr
        )

        self.activation = nn.GELU()
        self.dropout = (
            nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        )

        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
        """
        Args:
            u: (B, H, L) if transposed else (B, L, H)

        Returns:
            (B, H, L) output and None (dummy state placeholder).
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        k = self.kernel(L=L)  # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)  # (H, L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B, H, L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)

        y = y + u * self.D.unsqueeze(-1)  # D-term skip connection
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class S4Model(nn.Module):
    """Full S4D sequence model for regression / classification.

    Input:  (B, d_input, L)  — channels-first (aframe convention).
    Output: (B, d_output)
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList(
            [
                S4D(
                    d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    lr=lr,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout1d(dropout) for _ in range(n_layers)]
        )

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_input, L)

        Returns:
            (B, d_output)
        """
        x = x.transpose(-1, -2)  # (B, L, d_input)
        x = self.encoder(x)  # (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L)

        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts, strict=True
        ):
            z, _ = layer(x)
            z = dropout(z)
            x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)  # postnorm

        x = x.transpose(-1, -2)  # (B, L, d_model)
        x = x.mean(dim=1)  # (B, d_model) — pool over sequence
        return self.decoder(x)  # (B, d_output)
