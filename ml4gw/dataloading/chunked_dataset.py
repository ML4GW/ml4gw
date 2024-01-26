from collections.abc import Iterable

import torch


class ChunkedTimeSeriesDataset(torch.utils.data.IterableDataset):
    """
    Wrapper dataset that will loop through chunks of timeseries
    data produced by another iterable and sample windows from
    these chunks.

    Args:
        chunk_it:
            Iterator which will produce chunks of timeseries
            data to sample windows from. Should have shape
            `(N, C, T)`, where `N` is the number of chunks
            to sample from, `C` is the number of channels,
            and `T` is the number of samples along the
            time dimension for each chunk.
        kernel_size:
            Size of windows to be sampled from each chunk.
            Should be less than the size of each chunk
            along the time dimension.
        batch_size:
            Number of windows to sample at each iteration
        batches_per_chunk:
            Number of batches of windows to sample from
            each chunk before moving on to the next one.
            Sampling fewer batches from each chunk means
            a lower likelihood of sampling duplicate windows,
            but an increase in chunk-loading overhead.
        coincident:
            Whether the windows sampled from individual
            channels in each batch element should be
            sampled coincidentally, i.e. consisting of
            the same timesteps, or whether each window
            should be sample independently from the others.
        device:
            Which device chunks should be moved to upon loading.
    """

    def __init__(
        self,
        chunk_it: Iterable,
        kernel_size: float,
        batch_size: int,
        batches_per_chunk: int,
        coincident: bool = True,
        device: str = "cpu",
    ) -> None:
        self.chunk_it = chunk_it
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk
        self.coincident = coincident
        self.device = device

    def __len__(self):
        return len(self.chunk_it) * self.batches_per_chunk

    def __iter__(self):
        it = iter(self.chunk_it)
        chunk = next(it)
        num_chunks, num_channels, chunk_size = chunk.shape

        # if we're sampling coincidentally, we only need
        # to sample indices on a per-batch-element basis.
        # Otherwise, we'll need indices for both each
        # batch sample _and_ each channel with each sample
        if self.coincident:
            sample_size = (self.batch_size,)
        else:
            sample_size = (self.batch_size, num_channels)

        # slice kernels out a flattened chunk tensor
        # index-for-index. We'll account for batch/
        # channel indices by introducing offsets later on
        idx = torch.arange(self.kernel_size, device=self.device)
        idx = idx.view(1, 1, -1)
        idx = idx.repeat(self.batch_size, num_channels, 1)

        # this will just be a set of aranged channel indices
        # repeated to offset the kernel indices in the
        # flattened chunk tensor
        channel_idx = torch.arange(num_channels, device=self.device)
        channel_idx = channel_idx.view(1, -1, 1)
        channel_idx = channel_idx.repeat(self.batch_size, 1, self.kernel_size)
        idx += channel_idx * chunk_size

        while True:
            # record the number of rows in the chunk, then
            # flatten it to make it easier to slice
            if chunk_size < self.kernel_size:
                raise ValueError(
                    "Can't sample kernels of size {} from chunk "
                    "with size {}".format(self.kernel_size, chunk_size)
                )
            chunk = chunk.reshape(-1)

            # generate batches from the current chunk
            for _ in range(self.batches_per_chunk):
                # first sample the indices of which chunk elements
                # we're going to read batch elements from
                chunk_idx = torch.randint(
                    0, num_chunks, size=sample_size, device=self.device
                )

                # account for the offset this batch element
                # introduces in the flattened array
                chunk_idx *= num_channels * chunk_size
                chunk_idx = chunk_idx.view(self.batch_size, -1, 1)
                chunk_idx = chunk_idx + idx

                # now sample the start index within each chunk
                # element we're going to grab our time windows from
                time_idx = torch.randint(
                    0,
                    chunk_size - self.kernel_size,
                    size=sample_size,
                    device=self.device,
                )
                time_idx = time_idx.view(self.batch_size, -1, 1)

                # there's no additional offset factor to account for here
                chunk_idx += time_idx

                # now slice this 3D tensor from our flattened chunk
                yield chunk[chunk_idx]

            try:
                chunk = next(it)
            except StopIteration:
                break
            num_chunks, num_channels, chunk_size = chunk.shape
