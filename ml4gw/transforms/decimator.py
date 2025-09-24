import torch


class Decimator(torch.nn.Module):
    """
    Downsample/decimate timeseries according to a schedule.

    The schedule is a tensor of shape (N, 3), where each row is:
    [start_time, end_time, target_sample_rate]

    Example schedule (seconds, Hz):
    [[0, 40, 256],   # first 40s at 256 Hz
     [40, 58, 512],  # next 18s at 512 Hz
     [58, 60, 2048]] # last 2s at full 2048 Hz

    Strain: Tensor of shape (B, C, T), where
    B=batch size, C=channels, T=time.

    Functions:
        build_variable_indices(sample_rate, schedule, device)
            Compute the sample indices to keep from the full strain.
        split_by_schedule(strain, schedule)
            Split a strain into segments according to the schedule.
    """

    def __init__(
        self,
        sample_rate: int = None,
        schedule: torch.Tensor = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.schedule = schedule

        self._validate_inputs()
        idx = self.build_variable_indices()
        self.register_buffer("idx", idx)

    def _validate_inputs(
        self,
        strain: torch.Tensor = None,
    ):
        """
        Validate inputs for compatibility.
        If `strain` is None, only validates schedule and sample_rate.
        If `strain` is provided, validate only strain (called in forward).
        """
        if strain is None:
            if self.schedule.ndim != 2 or self.schedule.shape[1] != 3:
                raise ValueError(
                    f"Schedule must be of shape (N, 3), got \
                        {self.schedule.shape}"
                )

            if not torch.all(self.schedule[:, 1] > self.schedule[:, 0]):
                raise ValueError(
                    "Each schedule segment must have end_time > start_time"
                )

            if torch.any(self.sample_rate % self.schedule[:, 2].long() != 0):
                raise ValueError(
                    f"Sample rate {self.sample_rate} must be divisible by all "
                    f"target rates {self.schedule[:, 2].tolist()}"
                )

        else:
            if strain.ndim < 1:
                raise ValueError(
                    "strain must have at least 1 dimension (time axis)"
                )

            strain_len = int(
                (self.schedule[:, 1][-1] - self.schedule[:, 0][0])
                * self.sample_rate
            )
            if strain_len != strain.shape[-1]:
                raise ValueError(
                    f"Waveform length {strain.shape[-1]} does not match "
                    f"schedule duration {strain_len}"
                )

    def build_variable_indices(self):
        """
        Build a mask of indices to decimate a strain according to the schedule.
        """
        idx = torch.tensor([], dtype=torch.long)

        for s in self.schedule:
            if idx.size(0) == 0:
                start = int(s[0] * self.sample_rate)
            else:
                start = int(idx[-1]) + int(idx[-1] - idx[-2])
            stop = int(start + (s[1] - s[0]) * self.sample_rate)
            step = int(self.sample_rate // s[2])
            new_idx = torch.arange(start, stop, step, dtype=torch.long)
            idx = torch.cat((idx, new_idx))
        return idx

    def split_by_schedule(self, strain: torch.Tensor):
        """
        Split a strain into segments according to schedule.
        """
        split_sizes = (
            ((self.schedule[:, 1] - self.schedule[:, 0]) * self.schedule[:, 2])
            .long()
            .tolist()
        )
        segments = torch.split(strain, split_sizes, dim=-1)
        return segments

    def forward(
        self,
        strain: torch.Tensor,
        split: bool = False,
    ):
        """
        Run decimation: get decimated waveform and/or return split segments.

        Args:
            strain : torch.Tensor
                Input tensor of shape (B, C, T), where the last dimension
                is time.
            split : bool, optional
                If True, returns list of segments split by schedule along
                the last dim.

        Returns:
            torch.Tensor or List[torch.Tensor]
                Decimated strain or list of segments if split=True.
        """

        self._validate_inputs(strain=strain)
        dec_strain = strain.index_select(dim=-1, index=self.idx)

        if split:
            dec_strain = self.split_by_schedule(dec_strain)  # list of segments
        return dec_strain
