from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
import torch


@dataclass
class ChunkLoader:
    fnames: List[str]
    channels: List[str]
    chunk_size: int
    reads_per_chunk: int
    chunks_per_epoch: int
    coincident: bool = True

    def __post_init__(self):
        sizes = []
        for f in self.fnames:
            with h5py.File(f, "r") as f:
                size = len(f[self.channels[0]][:])
                sizes.append(size)
        total = sum(sizes)
        self.probs = np.array([i / total for i in sizes])

    def __len__(self):
        return self.chunks_per_epoch

    def sample_fnames(self):
        return np.random.choice(
            self.fnames,
            probs=self.probs,
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
                yield self.load_coincident()
            else:
                yield self.load_noncoincident()

    def __iter__(self):
        return self.iter_epoch()


class ChunkedDataloader(torch.utils.data.IterableDataset):
    """
    Iterable for generating batches of background data
    loaded on-the-fly from multiple HDF5 files. Loads
    `chunk_length`-sized randomly-sampled stretches of
    background from `reads_per_chunk` randomly sampled
    files up front, then samples `batches_per_chunk`
    batches of kernels from this chunk before loading
    in the next one. Terminates after `chunks_per_epoch`
    chunks have been exhausted, which amounts to
    `chunks_per_epoch * batches_per_chunk` batches.
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
    ) -> None:
        self.chunk_loader = ChunkLoader(
            fnames,
            channels,
            int(chunk_length * sample_rate),
            reads_per_chunk,
            chunks_per_epoch,
            coincident=coincident,
        )

        self.num_channels = len(channels)
        self.coincident = coincident
        self.kernel_size = int(kernel_length * sample_rate)
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self):
        return self.batches_per_chunk * len(self.chunk_loader)

    def iter_epoch(self):
        chunk_idx = np.zeros((self.batch_size, self.kernel_size))
        if not self.coincident:
            chunk_idx = np.repeat(
                chunk_idx[:, None], self.num_channels, axis=1
            )

        time_idx = np.arange(self.kernel_size)[None]
        time_idx = np.repeat(time_idx, self.batch_size, axis=0)
        if not self.coincident:
            time_idx = np.repeat(time_idx[:, None], self.num_channels, axis=1)

        channel_idx = np.arange(self.num_channels)[None, :, None]
        channel_idx = np.repeat(channel_idx, self.batch_size, axis=0)
        channel_idx = np.repeat(channel_idx, self.kernel_size, axis=2)

        for chunk in self.chunk_loader:
            for _ in range(self.batches_per_chunk):
                if self.coincident:
                    size = (self.batch_size,)
                else:
                    size = (self.batch_size, self.num_channels)

                idx = np.random.randint(len(chunk), size=size)[:, None]
                idx0 = chunk_idx + idx

                high = chunk.shape[-1] - idx.shape[-1]
                idx = np.random.randint(high, size=size)[:, None]
                idx2 = time_idx + idx[:, None]

                if self.coincident:
                    yield torch.Tensor(chunk[idx0, :, idx2])
                else:
                    yield torch.Tensor(chunk[idx0, channel_idx, idx2])

    def __iter__(self):
        return self.iter_epoch()
