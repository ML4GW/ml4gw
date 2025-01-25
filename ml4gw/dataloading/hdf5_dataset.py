import warnings
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch

from ml4gw.types import WaveformTensor


class ContiguousHdf5Warning(Warning):
    pass


class _Reader:
    def __new__(cls, fnames, path):
        if isinstance(fnames, str):
            cls = _SingleFileReader
        else:
            cls = _MultiFileReader
        return super().__new__(cls)

    def __init__(
        self, fnames: Union[str, Sequence[str]], path: Optional[str] = None
    ):
        self.fnames = fnames
        if path is not None:
            self.path = path.split("/")
        else:
            self.path = None
        self.sizes = {}

    def open(self, fname) -> tuple[h5py.File, h5py.Group]:
        f = group = h5py.File(fname, "r")
        if self.path is not None:
            for path in self.path:
                group = group[path]
        return f, group

    def _warn_non_contiguous(self, fname, dataset):
        warnings.warn(
            "File {} contains datasets at path {} that were generated "
            "without using chunked storage. This can have "
            "severe performance impacts at data loading time. "
            "If you need faster loading, try re-generating "
            "your datset with chunked storage turned on.".format(
                fname, "/".join(self.path) + "/" + dataset
            ),
            category=ContiguousHdf5Warning,
        )

    def get_sizes(self, channel):
        raise NotImplementedError

    def initialize_probs(self, channel):
        self.get_sizes(channel)
        total = sum(self.sizes.values())
        self.probs = np.array([self.sizes[k] / total for k in self.fnames])

    def sample_fnames(self, size):
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=size,
            replace=True,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        return

    @contextmanager
    def __call__(self, fname):
        raise NotImplementedError


class _MultiFileReader(_Reader):
    def get_sizes(self, channel):
        for fname in self.fnames:
            with self(fname) as f:
                dataset = f[channel]
                if dataset.chunks is None:
                    self._warn_non_contiguous(fname, channel)
                self.sizes[fname] = len(dataset)

    @contextmanager
    def __call__(self, fname):
        f, group = self.open(fname)
        with f:
            yield group


class _SingleFileReader(_Reader):
    _f = _group = None

    def get_sizes(self, channel):
        fname = self.fnames
        self.fname = fname
        with self:
            for key, group in self._group.items():
                dataset = group[channel]
                if dataset.chunks is None:
                    path = f"{key}/{channel}"
                    self._warn_non_contiguous(fname, path)
                self.sizes[key] = len(dataset)
            self.fnames = sorted(self._group.keys())

    def __enter__(self):
        self._f, self._group = self.open(self.fname)
        return self

    def __exit__(self, *exc_args):
        self._f.close()
        self._f = self._group = None

    @contextmanager
    def __call__(self, dataset):
        yield self._group[dataset]


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
        fnames: Union[Sequence[str], str],
        channels: Sequence[str],
        kernel_size: int,
        batch_size: int,
        batches_per_epoch: int,
        coincident: Union[bool, str],
        path: Optional[str] = None,
    ) -> None:
        if not isinstance(coincident, bool) and coincident != "files":
            raise ValueError(
                "coincident must be either a boolean or 'files', "
                "got unrecognized value {}".format(coincident)
            )

        self.reader = _Reader(fnames, path)
        self.reader.initialize_probs(channels[0])
        self.channels = channels
        self.num_channels = len(channels)
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.coincident = coincident

    def __len__(self) -> int:
        return self.batches_per_epoch

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
        fnames = self.reader.sample_fnames(size)

        unique_fnames, inv, counts = np.unique(
            fnames, return_inverse=True, return_counts=True
        )
        for i, (fname, count) in enumerate(zip(unique_fnames, counts)):
            size = self.reader.sizes[fname]
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
            with self.reader(fname) as f:
                for b, c, i in zip(batch_indices, channel_indices, idx):
                    x[b, c] = f[self.channels[c]][i : i + self.kernel_size]
        return torch.Tensor(x)

    def __iter__(self) -> WaveformTensor:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_batches = self.batches_per_epoch
        else:
            num_batches, remainder = divmod(
                self.batches_per_epoch, worker_info.num_workers
            )
            if worker_info.id < remainder:
                num_batches += 1

        with self.reader:
            for _ in range(num_batches):
                yield self.sample_batch()
