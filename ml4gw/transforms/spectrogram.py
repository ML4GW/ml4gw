import warnings
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


class MultiResolutionSpectrogram(torch.nn.Module):
    """
    Create a batch of multi-resolution spectrograms
    from a batch of timeseries. Input is expected to
    have the shape `(B, C, T)`, where `B` is the number
    of batches, `C` is the number of channels, and `T`
    is the number of time samples.

    For each timeseries, calculate multiple normalized
    spectrograms based on the `Spectrogram` `kwargs` given.
    Combine the spectrograms by taking the maximum value
    from the nearest time-frequncy bin.

    If the largest number of time bins among the spectrograms
    is `N` and the largest number of frequency bins is `M`,
    the output will have dimensions `(B, C, M, N)`

    Args:
        kernel_length:
            The length in seconds of the time dimension
            of the tensor that will be turned into a
            spectrogram
        sample_rate:
            The sample rate of the timeseries in Hz
        kwargs:
            Arguments passed in kwargs will used to create
            `torchaudio.transforms.Spectrogram`s. Each
            argument should be a list of values. Any list
            of length greater than 1 should be the same
            length
    """

    def __init__(
        self, kernel_length: float, sample_rate: float, **kwargs
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_length * sample_rate
        # This method of combination makes sense only when
        # the spectrograms are normalized, so enforce this
        if "normalized" in kwargs.keys():
            if not all(kwargs["normalized"]):
                raise ValueError(
                    "Received a value of False for 'normalized'. "
                    "This method of combination is sensible only for "
                    "normalized spectrograms."
                )
        else:
            kwargs["normalized"] = [True]
        self.kwargs = self._check_and_format_kwargs(kwargs)

        self.transforms = torch.nn.ModuleList(
            [Spectrogram(**k) for k in self.kwargs]
        )

        dummy_input = torch.ones(int(kernel_length * sample_rate))
        self.shapes = torch.tensor(
            [t(dummy_input).shape for t in self.transforms]
        )

        self.num_freqs = max([shape[0] for shape in self.shapes])
        self.num_times = max([shape[1] for shape in self.shapes])

        left_pad = torch.zeros(len(self.transforms), dtype=torch.int)
        top_pad = torch.zeros(len(self.transforms), dtype=torch.int)
        bottom_pad = torch.tensor(
            [int(self.num_freqs - shape[0]) for shape in self.shapes]
        )
        right_pad = torch.tensor(
            [int(self.num_times - shape[1]) for shape in self.shapes]
        )
        self.register_buffer("left_pad", left_pad)
        self.register_buffer("top_pad", top_pad)
        self.register_buffer("bottom_pad", bottom_pad)
        self.register_buffer("right_pad", right_pad)

        freq_idxs = torch.tensor(
            [
                [int(i * shape[0] / self.num_freqs) for shape in self.shapes]
                for i in range(self.num_freqs)
            ]
        )
        freq_idxs = freq_idxs.repeat(self.num_times, 1, 1).transpose(0, 1)
        time_idxs = torch.tensor(
            [
                [int(i * shape[1] / self.num_times) for shape in self.shapes]
                for i in range(self.num_times)
            ]
        )
        time_idxs = time_idxs.repeat(self.num_freqs, 1, 1)

        self.register_buffer("freq_idxs", freq_idxs)
        self.register_buffer("time_idxs", time_idxs)

    def _check_and_format_kwargs(self, kwargs: Dict[str, List]) -> List:
        lengths = sorted(set([len(v) for v in kwargs.values()]))

        if lengths[-1] > 3:
            warnings.warn(
                "Combining too many spectrograms can impede computation time. "
                "If performance is slower than desired, try reducing the "
                "number of spectrograms",
                RuntimeWarning,
            )

        if len(lengths) > 2 or (len(lengths) == 2 and lengths[0] != 1):
            raise ValueError(
                "Spectrogram keyword args should all have the same "
                f"length or be of length one. Got lengths {lengths}"
            )

        if len(lengths) == 2:
            size = lengths[1]
            kwargs = {k: v * int(size / len(v)) for k, v in kwargs.items()}

        return [dict(zip(kwargs, col)) for col in zip(*kwargs.values())]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate spectrograms of the input tensor and
        combine them into a single spectrogram

        Args:
            X:
                Batch of multichannel timeseries which will
                be used to calculate the multi-resolution
                spectrogram. Should have the shape
                `(B, C, T)`, where `B` is the number of
                batches, `C` is the  number of channels,
                and `T` is the number of time samples.
        """
        if X.shape[-1] != self.kernel_size:
            raise ValueError(
                "Expected time dimension to be "
                f"{self.kernel_size} samples long, got input with "
                f"{X.shape[-1]} samples"
            )

        spectrograms = [t(X) for t in self.transforms]

        padded_specs = []
        for spec, left, right, top, bottom in zip(
            spectrograms,
            self.left_pad,
            self.right_pad,
            self.top_pad,
            self.bottom_pad,
        ):
            padded_specs.append(F.pad(spec, (left, right, top, bottom)))

        padded_specs = torch.stack(padded_specs)
        remapped_specs = padded_specs[..., self.freq_idxs, self.time_idxs]
        remapped_specs = torch.diagonal(remapped_specs, dim1=0, dim2=-1)

        return torch.max(remapped_specs, axis=-1)[0]
