import torch
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


class MultiResolutionSpectrogram(torch.nn.Module):
    """
    Create a single spectrogram that combines information
    from multiple spectrograms of the same timeseries.
    Input is expected to have the shape `(B, C, T)`,
    where `B` is the number of batches, `C` is the number
    of channels, and `T` is the number of time samples.

    Given a list of `n_fft`s, calculate the spectrogram
    corresponding to each and combine them by taking the
    maximum value from each bin, which has been normalized.

    If the largest number of time bins among the spectrograms
    is `N` and the largest number of frequency bins is `M`,
    the output will have dimensions `(B, C, M, N)`
    """

    def __init__(
        self,
        n_ffts: torch.Tensor,
        sample_rate: float,
        kernel_length: float,
    ) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(
            [Spectrogram(n_fft, normalized=True) for n_fft in n_ffts]
        )

        dummy_input = torch.ones(kernel_length * sample_rate)
        dummy_shapes = torch.tensor(
            [t(dummy_input).shape for t in self.transforms]
        )
        self.register_buffer("shapes", dummy_shapes)

        self.num_freqs = max([shape[0] for shape in dummy_shapes])
        self.num_times = max([shape[1] for shape in dummy_shapes])

        freq_idxs = torch.tensor(
            [
                [int(i * shape[0] / self.num_freqs) for shape in dummy_shapes]
                for i in range(self.num_freqs)
            ]
        )
        freq_idxs = freq_idxs.repeat(self.num_times, 1, 1).transpose(0, 1)
        time_idxs = torch.tensor(
            [
                [int(i * shape[1] / self.num_times) for shape in dummy_shapes]
                for i in range(self.num_times)
            ]
        )
        time_idxs = time_idxs.repeat(self.num_freqs, 1, 1)

        self.register_buffer("freq_idxs", freq_idxs)
        self.register_buffer("time_idxs", time_idxs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        spectrograms = [t(X) for t in self.transforms]

        left = 0
        top = 0
        padded_specs = []
        for i, spec in enumerate(spectrograms):
            bottom = self.num_freqs - self.shapes[i][0]
            right = self.num_times - self.shapes[i][1]
            padded_specs.append(F.pad(spec, (left, right, top, bottom)))

        padded_specs = torch.stack(padded_specs)
        remapped_specs = padded_specs[..., self.freq_idxs, self.time_idxs]
        remapped_specs = torch.diagonal(remapped_specs, dim1=0, dim2=-1)

        return torch.max(remapped_specs, axis=-1)[0]
