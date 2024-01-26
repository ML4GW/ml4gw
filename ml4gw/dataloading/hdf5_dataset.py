import warnings
from typing import Sequence, Union

import h5py
import numpy as np
import torch

from ml4gw.types import WaveformTensor


class ContiguousHdf5Warning(Warning):
    pass


class Hdf5TimeSeriesDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset that samples and loads windows of
    timeseries data uniformly from a set of HDF5 files.
    It is _strongly_ recommended that these files have been
    written using [chunked storage]
    (https://docs.h5py.org/en/stable/high/dataset.html#chunked-storage).
    This has shown to produce increases in read-time speeds
    of over an order of magnitude.

    Args:
        fnames:
            Paths to HDF5 files from which to sample data.
        channels:
            Datasets to read from the indicated files, which
            will be stacked along dim 1 of the generated batches
            during iteration.
        kernel_size:
            Size of the windows to read, in number of samples.
            This will be the size of the last dimension of the
            generated batches.
        batch_size:
            Number of windows to sample at each iteration.
        batches_per_epoch:
            Number of batches to generate during each call
            to `__iter__`.
        coincident:
            Whether windows for each channel in a given batch
            element should be sampled coincidentally, i.e.
            corresponding to the same time indices from the
            same files, or should be sampled independently.
            For the latter case, users can either specify
            `False`, which will sample filenames independently
            for each channel, or `"files"`, which will sample
            windows independently within a given file for each
            channel. The latter setting limits the amount of
            entropy in the effective dataset, but can provide
            over 2x improvement in total throughput.
    """

    def __init__(
        self,
        fnames: Sequence[str],
        channels: Sequence[str],
        kernel_size: int,
        batch_size: int,
        batches_per_epoch: int,
        coincident: Union[bool, str],
    ) -> None:
        if not isinstance(coincident, bool) and coincident != "files":
            raise ValueError(
                "coincident must be either a boolean or 'files', "
                "got unrecognized value {}".format(coincident)
            )

        self.fnames = fnames
        self.channels = channels
        self.num_channels = len(channels)
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.coincident = coincident

        self.sizes = {}
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
        total = sum(self.sizes.values())
        self.probs = np.array([i / total for i in self.sizes.values()])

    def __len__(self) -> int:
        return self.batches_per_epoch

    def sample_fnames(self, size) -> np.ndarray:
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=size,
            replace=True,
        )

    def sample_batch(self) -> WaveformTensor:
        """
        Sample a single batch of multichannel timeseries
        """

        # allocate memory up front
        x = np.zeros((self.batch_size, len(self.channels), self.kernel_size))

        # sample filenames, but only loop through each unique
        # filename once to avoid unnecessary I/O overhead
        if self.coincident is not False:
            size = (self.batch_size,)
        else:
            size = (self.batch_size, self.num_channels)
        fnames = self.sample_fnames(size)

        unique_fnames, inv, counts = np.unique(
            fnames, return_inverse=True, return_counts=True
        )
        for i, (fname, count) in enumerate(zip(unique_fnames, counts)):
            size = self.sizes[fname]
            max_idx = size - self.kernel_size

            # figure out which batch indices should be
            # sampled from the current filename
            indices = np.where(inv == i)[0]

            # when sampling coincidentally either fully
            # or at the file level, all channels will
            # correspond to the same file
            if self.coincident is not False:
                batch_indices = np.repeat(indices, self.num_channels)
                channel_indices = np.arange(self.num_channels)
                channel_indices = np.concatenate([channel_indices] * count)
            else:
                batch_indices = indices // self.num_channels
                channel_indices = indices % self.num_channels

            # if we're sampling fully coincidentally, each
            # channel will be the same in each file
            if self.coincident is True:
                idx = np.random.randint(max_idx, size=count)
                idx = np.repeat(idx, self.num_channels)
            else:
                # otherwise, every channel will be different
                # for the given file
                idx = np.random.randint(max_idx, size=len(batch_indices))

            # open the file and sample a different set of
            # kernels for each batch element it occupies
            with h5py.File(fname, "r") as f:
                for b, c, i in zip(batch_indices, channel_indices, idx):
                    x[b, c] = f[self.channels[c]][i : i + self.kernel_size]
        return torch.Tensor(x)

    def __iter__(self) -> torch.Tensor:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_batches = self.batches_per_epoch
        else:
            num_batches, remainder = divmod(
                self.batches_per_epoch, worker_info.num_workers
            )
            if worker_info.id < remainder:
                num_batches += 1

        for _ in range(num_batches):
            yield self.sample_batch()
