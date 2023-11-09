from typing import Optional, Tuple

import torch

from ml4gw.utils.slicing import unfold_windows

Tensor = torch.Tensor


class OnlineAverager(torch.nn.Module):
    """
    Module for performing stateful online averaging of
    batches of overlapping timeseries. At present, the
    first `num_updates` predictions produced by this
    model will underestimate the true average.

    Args:
        update_size:
            The number of samples separating the timestamps
            of subsequent inputs.
        batch_size:
            The number of batched inputs to expect at inference
            time.
        num_updates:
            The number of steps over which to average predictions
            before returning them.
        num_channels:
            The expected channel dimension of the input passed
            to the module at inference time.
        offset:
            Number of samples to throw away from the front
            edge of the kernel when averaging.
    """

    def __init__(
        self,
        update_size: int,
        batch_size: int,
        num_updates: int,
        num_channels: int,
        offset: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.update_size = update_size
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.offset = offset

        # build a blank tensor into which we will embed
        # the updated snapshot predictions at the
        # appropriate time offset for in-batch averaging
        self.batch_update_size = int(batch_size * update_size)
        self.state_size = int((num_updates - 1) * update_size)
        blank_size = self.batch_update_size + self.state_size
        blank = torch.zeros((batch_size, num_channels, blank_size))
        self.register_buffer("blank", blank)

        # set up the indices at which the updated snapshots
        # will be embedded into the blank tensor
        idx = torch.arange(num_updates * update_size)
        idx = torch.stack([idx + i * update_size for i in range(batch_size)])
        idx = idx.view(batch_size, 1, -1).repeat(1, num_channels, 1)
        self.register_buffer("idx", idx)

        # normalization indices used to downweight the
        # existing average at each in-batch aggregation
        weights = torch.scatter(blank, -1, idx, 1).sum(0)
        weight_size = int(num_updates * update_size)
        weights = unfold_windows(weights, weight_size, update_size)
        self.register_buffer("weights", weights)

    def get_initial_state(self):
        return torch.zeros((self.num_channels, self.state_size))

    def forward(
        self, update: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.get_initial_state()

        # slice off the steps from this update closest
        # to the future that we'll actually use. Divide
        # these values by the number of updates up-front
        # for averaging purposes.
        start = -self.num_updates * self.update_size
        if self.offset is not None:
            end = -self.offset
            start += end
        else:
            end = None
        x = update[:, :, start:end] / self.num_updates

        # append zeros to the state into which we
        # can insert our updates
        state = torch.nn.functional.pad(state, (0, self.batch_update_size))

        # window the existing snapshot into overlapping
        # segments and average them with our new updates
        windowed = unfold_windows(state, x.size(-1), self.update_size)
        windowed /= self.weights
        windowed += x

        # embed these windowed averages into a blank
        # array with offsets so that we can add the
        # overlapping bits
        padded = torch.scatter(self.blank, -1, self.idx, windowed)
        new_state = padded.sum(axis=0)

        if self.num_updates == 1:
            # if we don't need stateful behavior,
            # just return the "snapshot" as-is
            output, new_state = new_state, self.get_initial_state()
        else:
            # otherwise split off the values that have finished
            # averaging and are being returned from the ones that
            # will comprise the snapshot at the next update
            splits = [self.batch_size, self.num_updates - 1]
            splits = [i * self.update_size for i in splits]
            output, new_state = torch.split(new_state, splits, dim=-1)
        return output, new_state
