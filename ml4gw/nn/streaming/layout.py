import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamLayout:
    """
    Batch layout of a stream of ``size`` samples.

    Args:
        size:
            Number of samples total in the stream.
        batch_size:
            The number of outputs to produce at each update.
        stride:
            The number of samples in between each output.
    """

    size: int
    batch_size: int
    stride: int

    @property
    def step_size(self) -> int:
        """Samples per batch update."""
        return self.batch_size * self.stride

    @property
    def num_batches(self) -> int:
        """Total number of batches in the stream, rounded up to include
        a zero-padded final batch.
        """
        return math.ceil(self.size / self.step_size)

    @property
    def remainder(self) -> int:
        """Samples in the final batch if ``step_size`` doesn't evenly
        divide ``size``, else 0."""
        return self.size % self.step_size

    @property
    def num_pad(self) -> int:
        """Padding samples appended to fill the final batch."""
        return self.num_batches * self.step_size - self.size

    @property
    def num_outputs(self) -> int:
        """Number of outputs, excluding those whose stride of new
        samples lies entirely in the padding of the final batch."""
        num_pad_outputs = self.num_pad // self.stride
        return self.num_batches * self.batch_size - num_pad_outputs

    def batch_bounds(self, batch_idx: int) -> tuple[int, int, bool]:
        """Sample range for one batch, and whether it is the last one.
        The final batch's range stops at the end of the unpadded stream."""
        last = batch_idx == self.num_batches - 1
        start = batch_idx * self.step_size
        end = start + self.step_size
        if last and self.remainder:
            end = start + self.remainder
        return start, end, last
