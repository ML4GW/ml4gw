from typing import Optional

import torch

from ml4gw.spectral import fast_spectral_density, spectral_density


class SpectralDensity(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        device: str = "cpu",
        fast: bool = False,
    ) -> None:
        if overlap is None:
            overlap = fftlength / 2
        elif overlap >= fftlength:
            raise ValueError(
                "Can't have overlap {} longer than fftlength {}".format(
                    overlap, fftlength
                )
            )

        super().__init__()

        self.nperseg = int(fftlength * sample_rate)
        self.nstride = self.nperseg - int(overlap * sample_rate)

        # do we allow for arbitrary windows?
        self.window = torch.hann_window(self.nperseg).to(device)

        # scale corresponds to "density" normalization, worth
        # considering adding this as a kwarg and changing this calc
        self.scale = 1.0 / (sample_rate * (self.window**2).sum())

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        self.device = device
        self.fast = fast

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.fast:
            return fast_spectral_density(
                x,
                y=y,
                nperseg=self.nperseg,
                nstride=self.nstride,
                window=self.window,
                scale=self.scale,
                average=self.average,
            )

        if y is not None:
            raise NotImplementedError
        return spectral_density(
            x,
            nperseg=self.nperseg,
            nstride=self.nstride,
            window=self.window,
            scale=self.scale,
            average=self.average,
        )
