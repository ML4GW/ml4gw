import torch


def get_freqs_from_time(
    window_length: float, sample_rate: float
) -> torch.Tensor:
    freqs = torch.arange((window_length * sample_rate) // 2 + 1)
    return freqs / window_length


def get_freqs_from_size(window_size: int, sample_rate: float) -> torch.Tensor:
    return torch.arange(window_size // 2 + 1) * sample_rate / window_size
