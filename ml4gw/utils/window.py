import torch


def unfold_windows(
    x: torch.Tensor,
    window_size: int,
    stride: int,
):
    """Unfold a timeseries into windows

    Args:
        x:
            The timeseries to unfold. Can have shape
            `(batch_size, num_channels, length * sample_rate)`,
            `(num_channels, length * sample_rate)`, or
            `(length * sample_rate)`
        window_size:
            The size of the windows to unfold from x
        stride:
            The stride between windows

    Returns:
       A tensor of shape
       `(batch_size, num_channels * num_windows, window_size)`
    """
    if x.ndim == 1:
        x = x[None, None, None, :]
    elif x.ndim == 2:
        x = x[None, :, None, :]
    elif x.ndim == 3:
        x = x[:, :, None, :]

    num_windows = (x.shape[-1] - window_size) // stride + 1
    stop = (num_windows - 1) * stride + window_size
    x = x[..., :stop]

    unfold_op = torch.nn.Unfold((1, num_windows), dilation=(1, stride))
    x = unfold_op(x)

    return x
