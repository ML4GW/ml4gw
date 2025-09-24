import torch


class Decimator(torch.nn.Module):
    """
    Downsample/decimate timeseries according to a schedule.

    The schedule is a tensor of shape (N, 3), where each row is:
    [start_time, end_time, target_sample_rate]

    Example default schedule (seconds, Hz):
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
        device=None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.schedule = schedule
        self.device = device

        if sample_rate is not None and schedule is not None:
            self._validate_inputs(sample_rate, schedule)
            self.idx = self.build_variable_indices(
                sample_rate, schedule, device
            )
        else:
            self.idx = None

    def _validate_inputs(
        self,
        sample_rate: int,
        schedule: torch.Tensor,
        strain: torch.Tensor = None,
    ):
        """
        Validate inputs for compatibility.
        If `strain` is None, only validates schedule and sample_rate.
        """
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(
                f"Sample rate must be a positive integer, got {sample_rate}"
            )

        if schedule.ndim != 2 or schedule.shape[1] != 3:
            raise ValueError(
                f"Schedule must be of shape (N, 3), got {schedule.shape}"
            )

        if not torch.all(schedule[:, 1] > schedule[:, 0]):
            raise ValueError(
                "Each schedule segment must have end_time > start_time"
            )

        if torch.any(schedule[:, 2] <= 0):
            raise ValueError("Target sample rates must be positive")

        if torch.any(sample_rate % schedule[:, 2].long() != 0):
            raise ValueError(
                f"Sample rate {sample_rate} must be divisible by all "
                f"target rates {schedule[:, 2].tolist()}"
            )

        if strain is not None:
            if strain.ndim < 1:
                raise ValueError(
                    "strain must have at least 1 dimension (time axis)"
                )

            expected_len = int(
                (schedule[:, 1][-1] - schedule[:, 0][0]) * sample_rate
            )
            if expected_len != strain.shape[-1]:
                raise ValueError(
                    f"Waveform length {strain.shape[-1]} does not match "
                    f"schedule duration {expected_len}"
                )

    def build_variable_indices(
        self,
        sample_rate: int = None,
        schedule: torch.Tensor = None,
        device=None,
    ):
        """
        Build a mask of indices to decimate a strain according to the schedule.
        """
        sample_rate = (
            sample_rate if sample_rate is not None else self.sample_rate
        )
        schedule = schedule if schedule is not None else self.schedule
        device = device if device is not None else self.device

        idx = torch.tensor([], dtype=torch.long, device=device)

        for s in schedule:
            if idx.size(0) == 0:
                start = int(s[0] * sample_rate)
            else:
                start = int(idx[-1]) + int(idx[-1] - idx[-2])
            stop = int(start + (s[1] - s[0]) * sample_rate)
            step = int(sample_rate // s[2])
            new_idx = torch.arange(
                start, stop, step, dtype=torch.long, device=device
            )
            idx = torch.cat((idx, new_idx))
        return idx

    def split_by_schedule(
        self, strain: torch.Tensor, schedule: torch.Tensor = None
    ):
        """
        Split a strain into segments according to schedule.
        """
        schedule = schedule if schedule is not None else self.schedule
        split_sizes = (
            ((schedule[:, 1] - schedule[:, 0]) * schedule[:, 2])
            .long()
            .tolist()
        )
        segments = torch.split(strain, split_sizes, dim=-1)
        return segments

    def forward(
        self,
        strain: torch.Tensor,
        sample_rate: int = None,
        schedule: torch.Tensor = None,
        split: bool = False,
        device=None,
    ):
        """
        Run decimation: get decimated waveform and/or return split segments.

        Parameters:
            strain : torch.Tensor
                Input tensor of shape (B, C, T), where the last dimension
                is time.
            sample_rate : int, optional
                Original sample rate of the strain in Hz. Defaults to
                self.sample_rate.
            schedule : torch.Tensor, optional
                Tensor of shape (N, 3), each row [start, end, target_sr].
            split : bool, optional
                If True, returns list of segments split by schedule along
                the last dim.
            device : torch.device, optional
                Device on which to perform indexing.

        Returns:
            torch.Tensor or List[torch.Tensor]
                Decimated strain or list of segments if split=True.
        """

        device = device if device is not None else self.device
        sample_rate = (
            sample_rate if sample_rate is not None else self.sample_rate
        )
        schedule = schedule if schedule is not None else self.schedule

        self._validate_inputs(sample_rate, schedule, strain=strain)

        if (
            self.idx is not None
            and sample_rate == self.sample_rate
            and torch.equal(schedule, self.schedule)
        ):
            idx = self.idx
        else:
            idx = self.build_variable_indices(
                sample_rate, schedule, device=device
            )

        dec_strain = strain.index_select(dim=-1, index=idx)

        if split:
            dec_strain = self.split_by_schedule(dec_strain, schedule=schedule)
            return dec_strain  # list of segments
        else:
            return dec_strain
