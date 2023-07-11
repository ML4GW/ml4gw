from typing import List

import h5py
import numpy as np
import torch


class ChunkLoader(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fnames: List[str],
        channels: List[str],
        chunk_size: int,
        reads_per_chunk: int,
        chunks_per_epoch: int,
        coincident: bool = True,
    ) -> None:
        self.fnames = fnames
        self.channels = channels
        self.chunk_size = chunk_size
        self.reads_per_chunk = reads_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.coincident = coincident

        sizes = []
        for f in self.fnames:
            with h5py.File(f, "r") as f:
                size = len(f[self.channels[0]])
                sizes.append(size)
        total = sum(sizes)
        self.probs = np.array([i / total for i in sizes])

    def sample_fnames(self):
        return np.random.choice(
            self.fnames,
            p=self.probs,
            size=(self.reads_per_chunk,),
            replace=True,
        )

    def load_coincident(self):
        fnames = self.sample_fnames()
        chunks = []
        for fname in fnames:
            with h5py.File(fname, "r") as f:
                chunk, idx = [], None
                for channel in self.channels:
                    if idx is None:
                        end = len(f[channel]) - self.chunk_size
                        idx = np.random.randint(0, end)
                    x = f[channel][idx : idx + self.chunk_size]
                    chunk.append(x)
            chunks.append(np.stack(chunk))
        return np.stack(chunks)

    def load_noncoincident(self):
        chunks = []
        for channel in self.channels:
            fnames = self.sample_fnames()
            chunk = []
            for fname in fnames:
                with h5py.File(fname, "r") as f:
                    end = len(f[channel]) - self.chunk_size
                    idx = np.random.randint(0, end)
                    x = f[channel][idx : idx + self.chunk_size]
                    chunk.append(x)
            chunks.append(np.stack(chunk))
        return np.stack(chunks, axis=1)

    def iter_epoch(self):
        for _ in range(self.chunks_per_epoch):
            if self.coincident:
                yield torch.Tensor(self.load_coincident())
            else:
                yield torch.Tensor(self.load_noncoincident())

    def collate(self, xs):
        return torch.cat(xs, axis=0)

    def __iter__(self):
        return self.iter_epoch()


class ChunkedDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset for generating batches of background data
    loaded on-the-fly from multiple HDF5 files. Loads
    `chunk_length`-sized randomly-sampled stretches of
    background from `reads_per_chunk` randomly sampled
    files up front, then samples `batches_per_chunk`
    batches of kernels from this chunk before loading
    in the next one. Terminates after `chunks_per_epoch`
    chunks have been exhausted, which amounts to
    `chunks_per_epoch * batches_per_chunk` batches.

    Note that filenames are not sampled uniformly
    at chunk-loading time, but are weighted according
    to the amount of data each file contains. This ensures
    a uniform sampling over time across the whole dataset.

    To load chunks asynchronously in the background,
    specify `num_workers > 0`. Note that if the
    number of workers is not an even multiple of
    `chunks_per_epoch`, the last chunks of an epoch
    will be composed of fewer than `reads_per_chunk`
    individual segments.

    Args:
        fnames:
            List of HDF5 archives containing data to read.
            Each file should have all of the channels specified
            in `channels` as top-level datasets.
        channels:
            Datasets to load from each filename in `fnames`
        kernel_length:
            Length of the windows returned at iteration time
            in seconds
        sample_rate:
            Rate at which the data in the specified `fnames`
            has been sampled.
        batch_size:
            Number of samples to return at iteration time
        reads_per_chunk:
            Number of file reads to perform when generating
            each chunk
        chunk_length:
            Amount of data to read for each segment loaded
            into each chunk, in seconds
        batches_per_chunk:
            Number of batches to sample from each chunk
            before loading the next one
        chunks_per_epoch:
            Number of chunks to generate before iteration
            terminates
        coincident:
            Flag indicating whether windows returned at iteration
            time should come from the same point in time for
            each channel in a given batch sample.
        num_workers:
            Number of workers for performing chunk loading
            asynchronously. If left as 0, chunk loading will
            be performed in serial with batch sampling.
        device:
            Device on which to host loaded chunks
    """

    def __init__(
        self,
        fnames: List[str],
        channels: List[str],
        kernel_length: float,
        sample_rate: float,
        batch_size: int,
        reads_per_chunk: int,
        chunk_length: float,
        batches_per_chunk: int,
        chunks_per_epoch: int,
        coincident: bool = True,
        num_workers: int = 0,
        device: str = "cpu",
    ) -> None:
        if not num_workers:
            reads_per_worker = reads_per_chunk
        else:
            reads_per_worker = int(reads_per_chunk // num_workers)

        chunk_loader = ChunkLoader(
            fnames,
            channels,
            int(chunk_length * sample_rate),
            reads_per_worker,
            chunks_per_epoch,
            coincident=coincident,
        )

        if not num_workers:
            self.chunk_loader = chunk_loader
        else:
            self.chunk_loader = torch.utils.data.DataLoader(
                chunk_loader,
                batch_size=num_workers,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=chunk_loader.collate,
            )

        self.device = device
        self.num_channels = len(channels)
        self.coincident = coincident

        self.chunk_size = int(chunk_length * sample_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk
        self.chunks_per_epoch = chunks_per_epoch
        self.num_workers = num_workers

    def __len__(self):
        if not self.num_workers:
            return self.chunks_per_epoch * self.batches_per_chunk

        num_chunks = (self.chunks_per_epoch - 1) // self.num_workers + 1
        return num_chunks * self.num_workers * self.batches_per_chunk

    def iter_epoch(self):
        # slice kernels out a flattened chunk tensor
        # index-for-index. We'll account for batch/
        # channel indices by introducing offsets later on
        idx = torch.arange(self.kernel_size, device=self.device)
        idx = idx.view(1, 1, -1)
        idx = idx.repeat(self.batch_size, self.num_channels, 1)

        # this will just be a set of aranged channel indices
        # repeated to offset the kernel indices in the
        # flattened chunk tensor
        channel_idx = torch.arange(self.num_channels, device=self.device)
        channel_idx = channel_idx.view(1, -1, 1)
        channel_idx = channel_idx.repeat(self.batch_size, 1, self.kernel_size)
        idx += channel_idx * self.chunk_size

        for chunk in self.chunk_loader:
            # record the number of rows in the chunk, then
            # flatten it to make it easier to slice
            num_chunks, _, chunk_size = chunk.shape
            chunk = chunk.to(self.device).reshape(-1)

            # generate batches from the current chunk
            for _ in range(self.batches_per_chunk):
                # if we're sampling coincidentally, we only need
                # to sample indices on a per-batch-element basis.
                # Otherwise, we'll need indices for both each
                # batch sample _and_ each channel with each sample
                if self.coincident:
                    size = (self.batch_size,)
                else:
                    size = (self.batch_size, self.num_channels)

                # first sample the indices of which chunk elements
                # we're going to read batch elements from
                chunk_idx = torch.randint(
                    0, num_chunks, size=size, device=self.device
                )

                # account for the offset this batch element
                # introduces in the flattened array
                chunk_idx *= self.num_channels * self.chunk_size
                chunk_idx = chunk_idx.view(self.batch_size, -1, 1)
                chunk_idx = chunk_idx + idx

                # now sample the start index within each chunk
                # element we're going to grab our time windows from
                time_idx = torch.randint(
                    0,
                    chunk_size - self.kernel_size,
                    size=size,
                    device=self.device,
                )
                time_idx = time_idx.view(self.batch_size, -1, 1)

                # there's no additional offset factor to account for here
                chunk_idx += time_idx

                # now slice this 3D tensor from our flattened chunk
                yield chunk[chunk_idx]

    def __iter__(self):
        return self.iter_epoch()
