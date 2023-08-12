import itertools
from typing import Optional, Tuple, Union

import torch

from ml4gw import types
from ml4gw.utils.slicing import slice_kernels


class InMemoryDataset:
    """Dataset for iterating through in-memory multi-channel timeseries

    Dataset for arrays of timeseries data which can be stored
    in-memory all at once. Iterates through the data by sampling
    fixed-length windows from all channels. The precise mechanism
    for this iteration is determined by combinations of the keyword
    arguments. See their descriptions for details.

    Args:
        X:
            Timeseries data to be iterated through. Should have
            shape `(num_channels, length * sample_rate)`. Windows
            will be sampled from the time (1st) dimension for all
            channels along the channel (0th) dimension.
        kernel_size:
            The length of the windows to sample from `X` in units
            of samples.
        y:
            Target timeseries to be iterated through. If specified,
            should be a single channel and have shape
            `(length * sample_rate,)`. If left as `None`, only windows
            sampled from `X` will be returned during iteration.
            Otherwise, windows sampled from both arrays will be
            returned. Note that if sampling is performed non-coincidentally,
            there's no sensible way to align windows sampled from this
            array with the windows sampled from `X`, so this combination
            of arguments is not permitted.
        batch_size:
            Maximum number of windows to return at each iteration. Will
            be the length of the 0th dimension of the returned array(s).
            If `batches_per_epoch` is specified, this will be the length
            of _every_ array returned during iteration. Otherwise, it's
            possible that the last array will be shorter due to the number
            of windows in the timeseries being a non-integer multiple of
            `batch_size`.
        stride:
            The resolution at which windows will be sampled from the
            specified timeseries, in units of samples. E.g. if
            `stride=2`, the first sample of each window can only be
            from an index of `X` which is a multiple of 2. Obviously,
            this reduces the number of windows which can be iterated
            through by a factor of `stride`.
        batches_per_epoch:
            Number of batches of window to produce during iteration
            before raising a `StopIteration`. Must be specified if
            performing non-coincident sampling. Otherwise, if left
            as `None`, windows will be sampled until the entire
            timeseries has been exhausted. Note that
            `batch_size * batches_per_epoch` must be be small
            enough to be able to be fulfilled by the number of
            windows in the timeseries, otherise a `ValueError`
            will be raised.
        coincident:
            Whether to sample windows from the channels of `X`
            using the same indices or independently. Can't be
            `True` if `batches_per_epoch` is `None` or `y` is
            _not_ `None`.
        shuffle:
            Whether to sample windows from timeseries randomly
            or in order along the time axis. If `coincident=False`
            and `shuffle=False`, channels will be iterated through
            with the index along the last channel moving fastest.
        device:
            Which device to host the timeseries arrays on
    """

    def __init__(
        self,
        X: types.TimeSeriesTensor,
        kernel_size: int,
        y: Optional[types.ScalarTensor] = None,
        batch_size: int = 32,
        stride: int = 1,
        batches_per_epoch: Optional[int] = None,
        coincident: bool = True,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        self.X = torch.Tensor(X).to(device)

        # make sure if we specified a target array that all other
        # other necessary conditions are met (it has the same
        # length as `X` and we're sampling coincidentally)
        if y is not None and y.shape[-1] != X.shape[-1]:
            raise ValueError(
                "Target timeseries must have same length as input"
            )
        elif y is not None and not coincident:
            raise ValueError("Can't sample target array non-coincidentally")
        elif y is not None:
            self.y = y.to(device)
        else:
            self.y = None

        if not coincident and batches_per_epoch is None:
            # TODO: do we want to allow this? There are strides
            # for which this might be pallatable, but deciding
            # where to draw the line feels more complicated than
            # just allowing the user to compute the appropriate
            # batches_per_epoch themselves if they so choose
            raise ValueError(
                "Must specify number of batches between validation "
                "steps for non-coincident sampling"
            )

        # make sure that we'll have enough kernels to be able
        # to generate the specified number of batches
        self.kernel_size = kernel_size
        self.stride = stride
        if batches_per_epoch is not None and self.num_kernels < (
            batch_size * batches_per_epoch
        ):
            raise ValueError(
                "Number of kernels {} in timeseries insufficient "
                "to generate {} batches of size {}".format(
                    self.num_kernels, batch_size, batches_per_epoch
                )
            )

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.shuffle = shuffle
        self.coincident = coincident
        self._i = self._idx = None

    @property
    def num_kernels(self) -> int:
        """
        The number of windows contained in the timeseries if we
        sample at the specified stride.
        """
        return (self.X.shape[-1] - self.kernel_size) // self.stride + 1

    def __len__(self) -> int:
        """
        The number of _batches_ contained in the timeseries
        """
        if self.batches_per_epoch is not None:
            return self.batches_per_epoch

        if self.coincident:
            return (self.num_kernels - 1) // self.batch_size + 1
        else:
            # TODO: this won't ever be triggered currently,
            # but leaving it in in case we ever choose to
            # support it
            num_kernels = self.num_kernels ** len(self.X)
            return (num_kernels - 1) // self.batch_size + 1

    def __iter__(self):
        """
        Initialize arrays of indices we'll use to slice
        through X and y at iteration time. This helps by
        taking care of building in any randomness upfront.
        """

        # establish how many kernels we'll actually be iterating through
        if self.batches_per_epoch is not None:
            num_kernels = self.batch_size * self.batches_per_epoch
        else:
            num_kernels = self.num_kernels

        # make sure the indices live on the same device as X (and y)
        device = self.X.device

        if not self.coincident and self.shuffle:
            # sampling non-coincidentally and randomly,
            # so generate sample indices for each channel
            # independently. In principle this means you
            # could sample the same set of kernels twice,
            # but for sensible numbers of kernels and channels
            # this likelihood is very low.
            idx = torch.randint(
                self.num_kernels,
                size=(num_kernels, len(self.X)),
                device=device,
            )
        elif not self.coincident:
            # sampling non-coincidentally but deterministically.
            # Use a little zip magic to only iterate through the
            # indices we'll need rather than having to generate
            # everything.
            idx = [range(self.num_kernels) for _ in range(len(self.X))]
            idx = zip(range(num_kernels), itertools.product(*idx))
            idx = torch.stack([torch.Tensor(i[1]) for i in idx])
            idx = idx.type(torch.int64).to(device)
        elif self.shuffle:
            # sampling randomly but coincidentally, so we
            # just need one set of indices and we can use
            # randperm to make sure we don't repeat
            idx = torch.randperm(self.num_kernels, device=device)
            idx = idx[:num_kernels]
        else:
            # the simplest case: deteriminstic and coincident
            idx = torch.arange(num_kernels, device=device)

        self._idx = idx
        self._i = 0
        return self

    def __next__(
        self,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._i is None or self._idx is None:
            raise TypeError(
                "Must initialize InMemoryDataset iteration "
                "before calling __next__"
            )

        # check if we're out of batches, and if so
        # make sure to reset before stopping
        if self._i >= len(self):
            self._i = self._idx = None
            raise StopIteration

        # slice the array of _indices_ we'll be using to
        # slice our timeseries, and scale them by the stride
        slc = slice(self._i * self.batch_size, (self._i + 1) * self.batch_size)
        idx = self._idx[slc] * self.stride

        # slice our timeseries
        X = slice_kernels(self.X, idx, self.kernel_size)
        if self.y is not None:
            y = slice_kernels(self.y, idx, self.kernel_size)

        self._i += 1
        if self.y is not None:
            return X, y
        return X
