import os
import warnings
from typing import Sequence, Union

import h5py
import numpy as np
import torch
from ml4gw.types import WaveformTensor


class ContiguousHdf5Warning(Warning):
    pass


def parse_gps_from_fname(fname: str):
    base = os.path.basename(fname).replace(".h5", "")
    _, gps_start, duration = base.split("-")
    return int(gps_start), int(duration)


class Hdf5TimeSeriesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fnames: Sequence[str],
        channels: Sequence[str],
        kernel_size: int,
        psd_length: float,     # seconds to reserve from the beginning of each kernel
        sample_rate: int,
        batch_size: int,
        batches_per_epoch: int,
        coincident: Union[bool, str],
        mode: str = "raw",  # "raw", "clean", or "glitch"
        glitch_root: str = "/home/hongyin.chen/anti_gravity/gwak/gwak/output/omicron/HL",
        ifos: Sequence[str] = ("H1", "L1"),
        glitch_margin: float = 2.0,  # seconds
    ) -> None:
        assert mode in ("raw", "clean", "glitch")
        if not isinstance(coincident, bool) and coincident != "files":
            raise ValueError(
                f"coincident must be a boolean or 'files', got {coincident}"
            )

        self.fnames = fnames
        self.channels = channels
        self.num_channels = len(channels)
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.coincident = coincident
        self.mode = mode
        self.glitch_root = glitch_root
        self.ifos = ifos
        self.glitch_margin = glitch_margin
        self.psd_length = psd_length

        self.sizes = {}
        self.valid_indices = {}

        for fname in self.fnames:
            gps_start, duration = parse_gps_from_fname(fname)

            with h5py.File(fname, "r") as f:
                dset = f[channels[0]]
                if dset.chunks is None:
                    warnings.warn(
                        f"File {fname} contains datasets without chunked storage. "
                        "This may impact I/O performance.",
                        category=ContiguousHdf5Warning,
                    )
                self.sizes[fname] = len(dset)

                if self.mode == "raw":
                    valid = np.arange(self.sizes[fname] - kernel_size)
                else:
                    glitch_path = os.path.join(
                        self.glitch_root, f"Segs_{gps_start}_{duration}", "glitch_info.h5"
                    )
                    glitch_times = []
                    if os.path.exists(glitch_path):
                        with h5py.File(glitch_path, "r") as gf:
                            for ifo in self.ifos:
                                glitch_times.extend(gf[ifo]["time"][:])
                        glitch_times = np.array(glitch_times)
                    else:
                        warnings.warn(f"Glitch file not found: {glitch_path}")
                        glitch_times = np.array([])

                    mask = np.zeros(self.sizes[fname], dtype=bool)

                    for t in glitch_times:
                        start = int((t - gps_start - glitch_margin) * self.sample_rate)
                        stop = int((t - gps_start + glitch_margin) * self.sample_rate)
                        mask[max(0, start):min(self.sizes[fname], stop)] = True

                    if self.mode == "clean":
                        mask = ~mask  # keep only clean regions

                    # Only apply masking to the portion after PSD
                    psd_samples = int(self.psd_length * self.sample_rate)
                    mask_start = self.kernel_size - psd_samples
                    valid_range = mask[mask_start:self.sizes[fname] - self.kernel_size]
                    valid = np.where(valid_range)[0] + mask_start

                self.valid_indices[fname] = valid

        total = sum(len(v) for v in self.valid_indices.values())
        self.probs = np.array([len(self.valid_indices[f]) / total for f in self.fnames])

    def __len__(self) -> int:
        return self.batches_per_epoch

    def sample_fnames(self, size) -> np.ndarray:
        return np.random.choice(self.fnames, p=self.probs, size=size, replace=True)

    def sample_batch(self) -> WaveformTensor:
        x = np.zeros((self.batch_size, self.num_channels, self.kernel_size))

        if self.coincident is not False:
            size = (self.batch_size,)
        else:
            size = (self.batch_size, self.num_channels)

        fnames = self.sample_fnames(size)
        unique_fnames, inv, counts = np.unique(
            fnames, return_inverse=True, return_counts=True
        )

        for i, (fname, count) in enumerate(zip(unique_fnames, counts)):
            valid = self.valid_indices[fname]
            if len(valid) == 0:
                continue

            indices = np.where(inv == i)[0]

            if self.coincident is not False:
                batch_indices = np.repeat(indices, self.num_channels)
                channel_indices = np.tile(np.arange(self.num_channels), count)
            else:
                batch_indices = indices // self.num_channels
                channel_indices = indices % self.num_channels

            if self.coincident is True:
                idx = np.random.choice(valid, size=count)
                idx = np.repeat(idx, self.num_channels)
            else:
                idx = np.random.choice(valid, size=len(batch_indices))

            with h5py.File(fname, "r") as f:
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

        for _ in range(num_batches):
            yield self.sample_batch()