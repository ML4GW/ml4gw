from typing import Optional, Sequence, Tuple

import torch

from ml4gw.utils.slicing import unfold_windows


class Snapshotter(torch.nn.Module):
    """
    Model for converting streaming state updates into
    a batch of overlapping snaphots of a multichannel
    timeseries. Can support multiple timeseries in a
    single state update via the `channels_per_snapshot`
    kwarg.

    Specifically, maps tensors of shape
    `(num_channels, batch_size * stride_size)` to a tensor
    of shape `(batch_size, num_channels, snapshot_size)`.
    If `channels_per_snapshot` is specified, it will return
    `len(channels_per_snapshot)` tensors of this shape,
    with the channel dimension replaced by the corresponding
    value of `channels_per_snapshot`. The last tensor returned
    at call time will be the current state that can be passed
    to the next `forward` call.

    Args:
        num_channels:
            Number of channels in the timeseries. If
            `channels_per_snapshot` is not `None`,
            this should be equal to `sum(channels_per_snapshot)`.
        snapshot_size:
            The size of the output snapshot windows in
            number of samples
        stride_size:
            The number of samples in between each output
            snapshot
        batch_size:
            The number of snapshots to produce at each
            update. The last dimension of the input
            tensor should have size `batch_size * stride_size`.
        channels_per_snapshot:
            How to split up the channels in the timeseries
            for different tensors. If left as `None`, all
            the channels will be returned in a single tensor.
            Otherwise, the channels will be split up into
            `len(channels_per_snapshot)` tensors, with each
            tensor's channel dimension being equal to the
            corresponding value in `channels_per_snapshot`.
            Therefore, if specified, these values should
            add up to `num_channels`.
    """

    def __init__(
        self,
        num_channels: int,
        snapshot_size: int,
        stride_size: int,
        batch_size: int,
        channels_per_snapshot: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if stride_size >= snapshot_size:
            raise ValueError(
                "Snapshotter can't accommodate stride {} "
                "which is greater than snapshot size {}".format(
                    stride_size, snapshot_size
                )
            )

        self.snapshot_size = snapshot_size
        self.stride_size = stride_size
        self.state_size = snapshot_size - stride_size
        self.batch_size = batch_size

        if channels_per_snapshot is not None:
            if sum(channels_per_snapshot) != num_channels:
                raise ValueError(
                    "Can't break {} channels into {}".format(
                        num_channels, channels_per_snapshot
                    )
                )
        self.channels_per_snapshot = channels_per_snapshot
        self.num_channels = num_channels

    def get_initial_state(self):
        return torch.zeros((self.num_channels, self.state_size))

    # TODO: use torchtyping annotations to make
    # clear what the expected shapes are
    def forward(
        self, update: torch.Tensor, snapshot: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        if snapshot is None:
            snapshot = self.get_initial_state()

        # append new data to the snapshot
        snapshot = torch.cat([snapshot, update], axis=-1)

        if self.batch_size > 1:
            snapshots = unfold_windows(
                snapshot, self.snapshot_size, self.stride_size
            )
        else:
            snapshots = snapshot[None]

        if self.channels_per_snapshot is not None:
            if snapshots.size(1) != self.num_channels:
                raise ValueError(
                    "Expected {} channels, found {}".format(
                        self.num_channels, snapshots.size(1)
                    )
                )
            snapshots = torch.split(
                snapshots, self.channels_per_snapshot, dim=1
            )
        else:
            snapshots = (snapshots,)

        # keep only the latest snapshot as our state
        snapshot = snapshot[:, -self.state_size :]
        return tuple(snapshots) + (snapshot,)
