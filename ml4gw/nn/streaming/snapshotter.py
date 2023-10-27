from collections.abc import Sequence
from typing import Optional

import torch

from ml4gw.utils.slicing import unfold_windows


class Snapshotter(torch.nn.Module):
    def __init__(
        self,
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
        self.channels_per_snapshot = channels_per_snapshot
        self.num_channels = sum(channels_per_snapshot)

    def get_initial_state(self):
        return torch.zeros((self.num_channels, self.state_size))

    # TODO: use torchtyping annotations to make
    # clear what the expected shapes are
    def forward(
        self,
        update: torch.Tensor,
        snapshot: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, ...]:
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
                        self.num_channels,
                        snapshots.size(1)
                    )
                )
            snapshots = torch.split(
                snapshots, self.channels_per_snapshot, dim=1
            )
        else:
            snapshots = (snapshots,)

        # keep only the latest snapshot as our state
        snapshot = snapshot[:, -self.state_size:]
        return tuple(snapshots) + (snapshot,)
