from typing import Callable, Optional

import numpy as np
import torch

from ml4gw import gw
from ml4gw.spectral import Background, normalize_psd
from ml4gw.transforms.transform import FittableTransform


class SnrRescaler(FittableTransform):
    def __init__(
        self,
        n_ifos: int,
        sample_rate: float,
        waveform_duration: float,
        highpass: Optional[float] = None,
        distribution: Optional[Callable] = None,
    ):
        super().__init__()
        self.distribution = distribution
        self.highpass = highpass
        self.sample_rate = sample_rate
        self.df = 1 / waveform_duration
        waveform_size = int(waveform_duration * sample_rate)
        num_freqs = int(waveform_size // 2 + 1)
        buff = torch.zeros((n_ifos, num_freqs), dtype=torch.float64)
        self.register_buffer("background", buff)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer("mask", freqs >= highpass, persistent=False)
        else:
            self.mask = None

    def fit(
        self,
        *backgrounds: Background,
        sample_rate: Optional[float] = None,
        fftlength: float = 2
    ):
        psds = []
        for background in backgrounds:
            psd = normalize_psd(
                background, self.df, self.sample_rate, sample_rate, fftlength
            )
            psds.append(psd)

        background = torch.tensor(np.stack(psds), dtype=torch.float64)
        super().build(background=background)

    def forward(self, responses: gw.WaveformTensor):
        snrs = gw.compute_network_snrs(
            responses, self.background, self.sample_rate, self.mask
        )
        if self.distribution is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]
        else:
            target_snrs = self.distribution(len(snrs), snrs.device)

        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs
