import os, warnings, h5py, numpy as np, torch
from typing import Optional, Sequence, Union
from ml4gw.types import WaveformTensor

class ContiguousHdf5Warning(Warning):
    pass


def gps_from_fname(fname: str) -> tuple[int, int]:
    """
    background-1403027575-9210.h5  ➜  (1403027575, 9210)
    """
    stem  = os.path.basename(fname).replace(".h5", "")
    _, g0, dur = stem.split("-")
    return int(g0), int(dur)


def read_glitch_times(glitch_file: str,
                      ifos: Sequence[str]) -> np.ndarray:
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
    *mode = "raw"*   : sample every possible window (baseline behaviour).

    *mode = "clean"* :   – sample exactly as raw **but**
                         – discard any window whose *post-PSD* segment
                           (length = fduration + kernel_length) overlaps
                           a glitch (±glitch_margin seconds).

    *mode = "glitch*: – pick one glitch time *tgps*
                       – start-index  s = (tgps - gps_start) * fs − psd_samples
                         so the glitch lands in the centre of the
                         fduration+kernel_length part.
    """

    # ----------  constructor  ------------------------------------------------
    def __init__(self,
                 fnames           : Sequence[str],
                 channels         : Sequence[str],
                 kernel_length    : float,          # sec of data after whiten
                 psd_length       : float,          # whitening segment [s]
                 fduration        : float,          # time-domain pad for FFT [s]
                 sample_rate      : int,            # Hz
                 batch_size       : int,
                 batches_per_epoch: int,
                 coincident       : Union[bool, str],
                 mode             : str = "raw",    # raw | clean | glitch
                 glitch_root      : str = "/path/to/omicron/HL",
                 ifos             : Sequence[str] = ("H1","L1"),
                 glitch_margin    : float = 2.0,    # sec on each side
                 num_files_per_batch: Optional[int] = None):

        assert mode in ("raw","clean","glitch")
        if not isinstance(coincident,bool) and coincident!="files":
            raise ValueError("coincident must be bool or 'files'")

        # save parameters
        self.fnames       = np.asarray(fnames)
        self.channels     = channels
        self.num_channels = len(channels)
        self.kernel_len_s = kernel_length
        self.psd_len_s    = psd_length
        self.fdur_s       = fduration
        self.fs           = sample_rate
        self.kernel_size  = int((psd_length + fduration + kernel_length)*sample_rate)
        self.psd_samples  = int(psd_length * sample_rate)
        self.post_len     = int((fduration + kernel_length) * sample_rate)
        self.batch_size   = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.coincident   = coincident
        self.mode         = mode
        self.glitch_root  = glitch_root
        self.ifos         = ifos
        self.glitch_margin= glitch_margin
        self.num_files_per_batch = (
            len(fnames) if num_files_per_batch is None else num_files_per_batch)

        self.sizes, self.valid = {}, {}
        for fname in self.fnames:
            with h5py.File(fname,"r") as f:
                dset = f[channels[0]]
                if dset.chunks is None:
                    warnings.warn(f"{fname} stored contiguously – slower I/O",
                                  ContiguousHdf5Warning, stacklevel=2)
                self.sizes[fname] = len(dset)

            if mode == "raw":
                self.valid[fname] = np.arange(self.sizes[fname]-self.kernel_size)
                continue

            g0, dur      = gps_from_fname(fname)
            glitch_file  = os.path.join(glitch_root,
                                        f"Segs_{g0}_{dur}",
                                        "glitch_info.h5")
            tglitch      = read_glitch_times(glitch_file, ifos)

            mask = np.zeros(self.sizes[fname], dtype=bool)
            if len(tglitch):
                for tg in tglitch:
                    lo = int((tg-g0-glitch_margin)*self.fs)
                    hi = int((tg-g0+glitch_margin)*self.fs)
                    mask[max(lo,0):min(hi,self.sizes[fname])] = True

            if mode == "clean":
                # keep indices i such that the *post-PSD* slice [i+psd : i+kernel]
                # does NOT overlap a glitch → need mask on that region to be false
                keep = []
                max_i = self.sizes[fname]-self.kernel_size
                for i in range(max_i):
                    post = mask[i+self.psd_samples : i+self.kernel_size]
                    if not post.any():
                        keep.append(i)
                self.valid[fname] = np.asarray(keep, dtype=int)

            else:
                starts = []
                for tg in tglitch:
                    centre = int((tg - g0)*self.fs)          # index of glitch
                    s      = centre - self.psd_samples       # window start
                    if 0 <= s <= self.sizes[fname]-self.kernel_size:
                        starts.append(s)
                self.valid[fname] = np.unique(starts)

            if mode=="glitch" and len(self.valid[fname])==0:
                warnings.warn(f"No usable glitch windows in {fname}")
        # drop any file with zero valid indices
        self.fnames = np.asarray([f for f in self.fnames if len(self.valid[f])])
        if len(self.fnames)==0:
            raise RuntimeError(f"No files contain '{mode}' windows!")

        total = sum(len(self.valid[f]) for f in self.fnames)
        self.probs = np.array([len(self.valid[f])/total for f in self.fnames])

    def __len__(self): return self.batches_per_epoch

    def sample_files(self,size):
        idx = np.random.choice(np.arange(len(self.fnames)),
                               size=self.num_files_per_batch,
                               replace=False, p=self.probs)
        subf = self.fnames[idx]; p = self.probs[idx]; p/=p.sum()
        return np.random.choice(subf, size=size, replace=True, p=p)

    def sample_batch(self)->WaveformTensor:
        x = np.zeros((self.batch_size,self.num_channels,self.kernel_size),
                     dtype=np.float32)

        size = (self.batch_size,) if self.coincident else \
               (self.batch_size,self.num_channels)
        files = self.sample_files(size)

        uniq,inv,count = np.unique(files,return_inverse=True,return_counts=True)
        for u,(fname,cnt) in enumerate(zip(uniq,count)):
            inds = np.where(inv==u)[0]
            if self.coincident:
                b_idx = np.repeat(inds,self.num_channels)
                ch_idx= np.tile(np.arange(self.num_channels),cnt)
            else:
                b_idx = inds//self.num_channels
                ch_idx= inds%self.num_channels

            if self.coincident is True:
                starts = np.random.choice(self.valid[fname], size=cnt)
                starts = np.repeat(starts,self.num_channels)
            else:
                starts = np.random.choice(self.valid[fname], size=len(b_idx))

            with h5py.File(fname,"r") as f:
                for b,c,s in zip(b_idx,ch_idx,starts):
                    x[b,c] = f[self.channels[c]][s:s+self.kernel_size]

        return torch.tensor(x)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        n   = self.batches_per_epoch if wi is None else \
              self.batches_per_epoch//wi.num_workers + (wi.id < self.batches_per_epoch%wi.num_workers)
        for _ in range(n):
            yield self.sample_batch()