import torch


class Decimator(torch.nn.Module):
    r"""
    Downsample (decimate) a timeseries according to a user-defined schedule.

    .. note::

        This is a naive decimator that does not use any IIR/FIR filtering
        and selects every M-th sample according to the schedule.

    The schedule specifies which segments of the input to keep and at what
    sampling rate. Each row of the schedule has the form:

        `[start_time, end_time, target_sample_rate]`

    Args:
        sample_rate (int):
            Sampling rate (Hz) of the input timeseries.
        schedule (torch.Tensor):
            Tensor of shape `(N, 3)` defining start time, end time,
            and target sample rate for each segment.

    Shape:
        - Input: `(B, C, T)` where
            - B = batch size
            - C = channels
            - T = number of timesteps
                    (must equal schedule duration × sample_rate)
        - Output:
            - If ``split=False`` → `(B, C, T')` where `T'` is total
              number of decimated samples across all segments.
            - If ``split=True`` → list of tensors, one per segment.

    Returns:
        torch.Tensor or List[torch.Tensor]:
            The decimated timeseries, or list of decimated segments if
            ``split=True``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from ml4gw.transforms.decimator import Decimator

            >>> sample_rate = 2048
            >>> X_duration = 60

            >>> schedule = torch.tensor(
            ...     [[0, 40, 256], [40, 58, 512], [58, 60, 2048]],
            ...     dtype=torch.int,
            ... )

            >>> decimator = Decimator(sample_rate=sample_rate,
            ...    schedule=schedule)
            >>> X = torch.randn(1, 1, sample_rate * X_duration)
            >>> X_dec = decimator(X)
            >>> X_seg = decimator(X, split=True)

            >>> print("Original shape:", X.shape)
            Original shape: torch.Size([1, 1, 122880])
            >>> print("Decimated shape:", X_dec.shape)
            Decimated shape: torch.Size([1, 1, 23552])
            >>> for i, seg in enumerate(X_seg):
            ...     print(f"Segment {i} shape:", seg.shape)
            Segment 0 shape: torch.Size([1, 1, 10240])
            Segment 1 shape: torch.Size([1, 1, 9216])
            Segment 2 shape: torch.Size([1, 1, 4096])
    """

    def __init__(
        self,
        sample_rate: int = None,
        schedule: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.schedule = schedule

        self._validate_inputs()
        idx = self.build_variable_indices()
        self.register_buffer("idx", idx)

        self.expected_len = int(
            (self.schedule[:, 1][-1] - self.schedule[:, 0][0])
            * self.sample_rate
        )

    def _validate_inputs(self) -> None:
        r"""
        Validate the schedule and sample_rate.
        """
        if self.schedule.ndim != 2 or self.schedule.shape[1] != 3:
            raise ValueError(
                f"Schedule must be of shape (N, 3), got {self.schedule.shape}"
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

    def build_variable_indices(self) -> torch.Tensor:
        r"""
        Compute the time indices to keep based on the schedule.

        Returns:
            torch.Tensor:
                1D tensor of indices used to decimate the input.
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

    def split_by_schedule(self, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
        r"""
        Split a decimated timeseries into segments according to the schedule.

        Args:
            X (torch.Tensor):
                Decimated input of shape `(B, C, T')`.

        Returns:
            tuple of torch.Tensor:
                Each segment has shape :math:`(B, C, T_i)`
                where :math:`T_i` is the length implied by
                the corresponding schedule row.
        """
        split_sizes = (
            ((self.schedule[:, 1] - self.schedule[:, 0]) * self.schedule[:, 2])
            .long()
            .tolist()
        )

        return torch.split(X, split_sizes, dim=-1)

    def forward(
        self,
        X: torch.Tensor,
        split: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        r"""
        Apply decimation to the input timeseries.

        Args:
            X (torch.Tensor):
                Input tensor of shape `(B, C, T)`, where `T` must equal
                schedule duration × sample_rate.
            split (bool, optional):
                If True, return a list of segments instead of a single
                concatenated tensor. Default: False.

        Returns:
            torch.Tensor or List[torch.Tensor]:
                Decimated timeseries, or list of decimated segments.
        """
        if X.shape[-1] != self.expected_len:
            raise ValueError(
                f"X length {X.shape[-1]} does not match "
                f"expected schedule duration {self.expected_len}"
            )

        X_dec = X.index_select(dim=-1, index=self.idx)

        if split:
            X_dec = self.split_by_schedule(X_dec)
        return X_dec
