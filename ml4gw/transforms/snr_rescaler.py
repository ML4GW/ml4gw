import torch

from ..gw import compute_network_snr
from ..types import BatchTensor, TimeSeries2d, WaveformTensor
from .transform import FittableSpectralTransform


class SnrRescaler(FittableSpectralTransform):
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        waveform_duration: float,
        highpass: float | None = None,
        lowpass: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        waveform_size = int(waveform_duration * sample_rate)
        num_freqs = int(waveform_size // 2 + 1)

        buff = torch.zeros((num_channels, num_freqs), dtype=dtype)
        self.register_buffer("background", buff)

        if highpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer(
                "highpass_mask", freqs >= highpass, persistent=False
            )
        else:
            self.highpass_mask = None
        if lowpass is not None:
            freqs = torch.fft.rfftfreq(waveform_size, 1 / sample_rate)
            self.register_buffer(
                "lowpass_mask", freqs < lowpass, persistent=False
            )
        else:
            self.lowpass_mask = None

    def fit(
        self,
        *background: TimeSeries2d,
        fftlength: float | None = None,
        overlap: float | None = None,
    ):
        if len(background) != self.num_channels:
            raise ValueError(
                f"Expected to fit whitening transform on {self.num_channels} "
                f"background timeseries, but was passed {len(background)}"
            )

        num_freqs = self.background.size(1)
        psds = []
        for x in background:
            psd = self.normalize_psd(
                x, self.sample_rate, num_freqs, fftlength, overlap
            )
            psds.append(psd)
        background = torch.stack(psds)
        super().build(background=background)

    def forward(
        self,
        responses: WaveformTensor,
        target_snrs: BatchTensor | None = None,
    ):
        snrs = compute_network_snr(
            responses,
            self.background,
            self.sample_rate,
            self.highpass_mask,
            self.lowpass_mask,
        )
        if target_snrs is None:
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]

        weights = target_snrs / snrs
        rescaled_responses = responses * weights.view(-1, 1, 1)

        return rescaled_responses, target_snrs
