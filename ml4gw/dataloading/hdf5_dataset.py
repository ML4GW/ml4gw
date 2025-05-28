import os
import warnings
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch

from ..types import WaveformTensor


class ContiguousHdf5Warning(Warning):
    pass


def gps_from_fname(fname: str) -> tuple[int, int]:
    """
    background-1403027575-9210.h5  ➔  (1403027575, 9210)
    """
    stem = os.path.basename(fname).replace(".h5", "")
    _, g0, dur = stem.split("-")
    return int(g0), int(dur)


def read_glitch_times(glitch_file: str, ifos: Sequence[str]) -> np.ndarray:
    """
    Returns an array (possibly empty) with all Omicron trigger gps times
    stored under <ifo>/time in the given *glitch_file*.
    """
    if not os.path.exists(glitch_file):
        return np.array([])
    ts = []
    with h5py.File(glitch_file, "r") as f:
        for ifo in ifos:
            if ifo in f and "time" in f[ifo]:
                ts.extend(f[ifo]["time"][:])
    return np.asarray(ts, dtype=float)


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
        kernel_length:
            Seconds of data after whitening
        psd_length:
            whitening segment [s]
        fduration:
            time-domain pad for FFT [s]
        sample_rate:
            sample rate in Hz
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
        num_files_per_batch:
            The number of unique files from which to sample
            batch elements each epoch. If left as `None`,
            will use all available files. Useful when reading
            from many files is bottlenecking dataloading.

        mode:
            "raw" -- sample every possible window (baseline behaviour).
            "clean" –- sample exactly as raw **but**
                    -– discard any window whose *post-PSD* segment
                       (length = fduration + kernel_length) overlaps
                       a glitch (±glitch_margin seconds).

            "glitch" -– pick one glitch time *tgps*
                     -– start-index  s = (tgps - gps_start) * fs − psd_samples
                        so the glitch lands in the centre of the
                        fduration+kernel_length part.
    """
    def __init__(self,
                 fnames: Sequence[str],
                 channels: Sequence[str],
                 kernel_length: float,
                 psd_length: float,
                 fduration: float,
                 sample_rate: int,
                 batch_size: int,
                 batches_per_epoch: int,
                 coincident: Union[bool, str],
                 mode: str = "raw",
                 glitch_root: str = "/path/to/omicron/HL",
                 ifos: Sequence[str] = ("H1", "L1"),
                 glitch_margin: float = 2.0,
                 num_files_per_batch: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 remake_cache: bool = False):

        assert mode in ("raw", "clean", "glitch")
        if not isinstance(coincident, bool) and coincident != "files":
            raise ValueError("coincident must be bool or 'files'")

        self.fnames = np.asarray(fnames)
        self.channels = channels
        self.num_channels = len(channels)
        self.kernel_len_s = kernel_length
        self.psd_len_s = psd_length
        self.fdur_s = fduration
        self.fs = sample_rate

        # New logic: psd_length | fdur/2 | kernel_length | fdur/2
        self.fdur_samples = int((fduration / 2) * sample_rate)
        self.psd_samples = int(psd_length * sample_rate)
        self.kernel_samples = int(kernel_length * sample_rate)
        self.kernel_size = self.psd_samples + 2 * self.fdur_samples + self.kernel_samples

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.coincident = coincident
        self.mode = mode
        self.glitch_root = glitch_root
        self.ifos = ifos
        self.glitch_margin = glitch_margin
        self.num_files_per_batch = len(fnames) if num_files_per_batch is None else num_files_per_batch
        self.cache_dir = cache_dir
        self.remake_cache = remake_cache

        self.sizes, self.valid, self.cache_paths, self.num_valid = {}, {}, {}, {}
        for fname in self.fnames:
            basename = os.path.basename(fname).replace(".h5", "")
            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.cache_dir, f"{basename}_{mode}_valid.npy")
            else:
                cache_path = os.path.join(os.path.dirname(fname), f"{basename}_{mode}_valid.npy")
            
            self.cache_paths[fname] = str(cache_path)
            self.num_valid[fname] = 0

            with h5py.File(fname, "r") as f:
                dset = f[channels[0]]
                if dset.chunks is None:
                    warnings.warn(f"{fname} stored contiguously – slower I/O", ContiguousHdf5Warning)
                self.sizes[fname] = len(dset)

            if os.path.exists(cache_path) and not self.remake_cache and mode != "raw":
                #self.valid[fname] = np.load(cache_path)
                valid = np.load(cache_path)
                self.num_valid[fname] = len(valid)
                self.valid[fname] = valid
                continue

            if mode == "raw":
                self.num_valid[fname] = self.sizes[fname] - self.kernel_size
                continue

            g0, dur = gps_from_fname(fname)
            glitch_file = os.path.join(glitch_root, f"Segs_{g0}_{dur}", "glitch_info.h5")
            tglitch = read_glitch_times(glitch_file, ifos)

            mask = np.zeros(self.sizes[fname], dtype=bool)
            if len(tglitch):
                for tg in tglitch:
                    lo = int((tg - g0 - glitch_margin) * self.fs)
                    hi = int((tg - g0 + glitch_margin) * self.fs)
                    mask[max(lo, 0):min(hi, self.sizes[fname])] = True

            if mode == "clean":
                keep = []
                max_i = self.sizes[fname] - self.kernel_size
                for i in range(max_i):
                    post = mask[i + self.psd_samples : i + self.psd_samples + self.fdur_samples + self.kernel_samples + self.fdur_samples]
                    if not post.any():
                        keep.append(i)
                self.valid[fname] = np.asarray(keep, dtype=int)
                #valid = np.asarray(keep, dtype=int)
            else:  # glitch mode
                starts = []
                for tg in tglitch:
                    centre = int((tg - g0) * self.fs)
                    mid_kernel_offset = self.psd_samples + self.fdur_samples + self.kernel_samples // 2
                    s = centre - mid_kernel_offset
                    if 0 <= s <= self.sizes[fname] - self.kernel_size:
                        starts.append(s)
                self.valid[fname] = np.unique(starts)
                #valid = np.unique(starts)

            if mode == "glitch" and len(self.valid[fname]) == 0:
                warnings.warn(f"No usable glitch windows in {fname}")

            np.save(cache_path, self.valid[fname])
            self.num_valid[fname] = len(self.valid[fname])

        self.fnames = np.asarray([f for f in self.fnames if self.num_valid[f] > 0])
        self.num_files_per_batch = min(self.num_files_per_batch, len(self.fnames))
        if len(self.fnames) == 0:
            raise RuntimeError(f"No files contain {mode} windows!")

        total = sum(self.num_valid[f] for f in self.fnames)
        self.probs = np.array([self.num_valid[f] / total for f in self.fnames])

    def __len__(self):
        return self.batches_per_epoch

    def sample_files(self, size):
        idx = np.random.choice(np.arange(len(self.fnames)),
                               size=self.num_files_per_batch,
                               replace=False, p=self.probs)
        subf = self.fnames[idx]
        p = self.probs[idx]
        p /= p.sum()
        return np.random.choice(subf, size=size, replace=True, p=p)

    def sample_batch(self) -> WaveformTensor:
        x = np.zeros((self.batch_size, self.num_channels, self.kernel_size), dtype=np.float32)

        size = (self.batch_size,) if self.coincident else (self.batch_size, self.num_channels)
        files = self.sample_files(size)

        uniq, inv, count = np.unique(files, return_inverse=True, return_counts=True)
        for u, (fname, cnt) in enumerate(zip(uniq, count)):
            if self.mode == "raw":
                # just need to sample a number from 0 to self.sizes[fname] - self.kernel_size
                valid_cache = self.sizes[fname] - self.kernel_size
            else:
                valid_cache = self.valid[fname]
            inds = np.where(inv == u)[0]
            if self.coincident:
                b_idx = np.repeat(inds, self.num_channels)
                ch_idx = np.tile(np.arange(self.num_channels), cnt)
            else:
                b_idx = inds // self.num_channels
                ch_idx = inds % self.num_channels

            if self.coincident is True:
                #starts = np.random.choice(self.valid[fname], size=cnt)
                starts = np.random.choice(valid_cache, size=cnt)
                starts = np.repeat(starts, self.num_channels)
            else:
                #starts = np.random.choice(self.valid[fname], size=len(b_idx))
                starts = np.random.choice(valid_cache, size=len(b_idx))

            with h5py.File(fname, "r") as f:
                for b, c, s in zip(b_idx, ch_idx, starts):
                    x[b, c] = f[self.channels[c]][s:s + self.kernel_size]

        return torch.tensor(x)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        n = self.batches_per_epoch if wi is None else \
            self.batches_per_epoch // wi.num_workers + (wi.id < self.batches_per_epoch % wi.num_workers)
        for _ in range(n):
            yield self.sample_batch()
