import math
import warnings
from typing import Optional, Sequence

import h5py
import numpy as np
import torch


class ContiguousHdf5Warning(Warning):
    pass


class InferenceDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fnames: Sequence[str],
        channels: Sequence[str],
        stride_size: int,
        shift_sizes: Optional[Sequence[int]] = None,
    ):
        """
        Chronologically load streaming updates from a set of HDF5 files
        corresponding to segments of data. Optionally provide shift_sizes
        that will shift data by corresponding amount for timeslides.

        Args:
            fnames:
                List of HDF5 files to load data from
            channels:
                List of channels to load from each HDF5 file
            stride_size:
                Number of samples to stride over for each update
            batch_size:
                Number of updates to include in each batch
            kernel_size:
                Number of samples to include in each update
            shift_sizes:
                List of shift sizes to apply to each channel

        """

        self.fnames = fnames
        self.stride_size = stride_size
        self.channels = channels

        if shift_sizes is not None:
            if len(shift_sizes) != len(channels):
                raise ValueError("Shifts must be the same length as channels")
        self.shift_sizes = shift_sizes or [0] * len(channels)

        self.sizes = {}
        self.lengths = {}
        for fname in self.fnames:
            with h5py.File(fname, "r") as f:
                dset = f[channels[0]]
                if dset.chunks is None:
                    warnings.warn(
                        "File {} contains datasets that were generated "
                        "without using chunked storage. This can have "
                        "severe performance impacts at data loading time. "
                        "If you need faster loading, try re-generating "
                        "your datset with chunked storage turned on.".format(
                            fname
                        ),
                        category=ContiguousHdf5Warning,
                    )
                self.sizes[fname] = len(dset)
                self.lengths[fname] = math.ceil(len(dset) // self.stride_size)

    @property
    def max_shift(self):
        return max(self.shift_sizes)

    def __len__(self):
        return sum(self.lengths.values())

    def __iter__(self):
        for fname in self.fnames:
            with h5py.File(fname, "r") as f:
                size = self.sizes[fname] - self.max_shift
                idx = 0
                while idx < size:
                    data = []
                    for channel, shift in zip(self.channels, self.shift_sizes):
                        start = idx + shift
                        stop = start + self.stride_size

                        # make sure that segments with shifts shorter
                        # than the max shift get their ends cut off
                        stop = min(size + shift, stop)
                        x = f[channel][start:stop]
                        data.append(x)

                    yield np.stack(data)
                    idx += self.stride_size
