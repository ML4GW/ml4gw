import torch

class Decimator(torch.nn.Module):
    """
    Downsample/decimate timeseries according to a schedule.

    The schedule is a tensor of shape (num_segments, 3), where each row is:
    [start_time, end_time, target_sample_rate]

    Example default schedule (seconds, Hz):
    [[0, 40, 256],     # first 40s at 256 Hz
     [40, 58, 512],    # next 18s at 512 Hz
     [58, 60, 2048]]   # last 2s at full 2048 Hz

    Functions:
    build_variable_indices(sr, schedule, device):
    Compute the sample indices to keep from the full signal.

    split_by_schedule(signal, schedule, device):
    Split a signal into segments according to the schedule.
    """

    def __init__(self):
        super().__init__()

    def build_variable_indices(
        self, 
        sr: int, 
        schedule: torch.Tensor, 
        device=None
    ):
        """
        Build a mask of indices to decimate a signal according to the schedule.
        """
        idx = torch.tensor([], dtype=torch.long, device=device)

        for s in schedule:
            if idx.size(0) == 0:
                start = int(s[0] * sr)
            else:
                # extrapolate next start from last step size
                start = int(idx[-1]) + int(idx[-1] - idx[-2])
            stop = int(start + (s[1] - s[0]) * sr)
            step = int(sr // s[2])
            idx = torch.cat((idx, torch.arange(start, stop, step, dtype=torch.long, device=device)))
        return idx

    def split_by_schedule(
        self, 
        signal: torch.Tensor, 
        schedule: torch.Tensor, 
    ):
        """
        Split a signal into segments according to schedule.
        """
        split = schedule[:, 2].unsqueeze(-1) * (schedule[:, 1].unsqueeze(-1) - schedule[:, 0].unsqueeze(-1))
        split_sizes = split.squeeze().tolist()
        d1 = torch.split(signal, split_sizes, dim=-1)
        return d1

    def forward(
        self,
        sr: int,
        signal: torch.Tensor,
        schedule: torch.Tensor,
        split: bool = False,
        device=None
    ):
        """
        Run decimation: get decimated waveform and/or return split segments.
        """
        idx = self.build_variable_indices(sr=sr, schedule=schedule, device=device)
        signal = signal.index_select(dim=-1, index=idx)

        if split:
            signal = self.split_by_schedule(signal, schedule=schedule, device=signal.device)
            return signal # list of segments
        else:
            return signal