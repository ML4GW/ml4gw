import torch
from jaxtyping import Float, Int
from torch import Tensor

from .buffer import StreamBuffer
from .layout import StreamLayout
from .whitener import StreamWhitener

StreamOutput = Float[Tensor, "stream kernel *output"]


class StreamRunner(torch.nn.Module):
    """
    Stream a batch of updates through a whitener and a model.

    Args:
        whitener:
            :class:`~ml4gw.nn.streaming.StreamWhitener` that whitens
            kernels and defines the structure of the
            :class:`~ml4gw.nn.streaming.StreamBuffer` that holds the
            data for PSD estimation and whitening. Each update
            must contain ``batch_size * stride_size`` new samples per
            stream. A subclass whose ``forward`` returns a tuple, for
            example kernels and features computed from the ASD, passes
            the tuple to the model as separate arguments.
        model:
            Model applied to the whitened kernels, or to the output
            of ``transform``. It may return a tensor or a tuple of
            tensors, each with ``streams * batch_size`` rows.
        transform:
            Optional module called as ``transform(kernels)`` before
            the model, for example to compute features from the
            kernels. It may return a tensor or a tuple of tensors,
            which are passed to the model as separate arguments. If
            left as ``None``, the model receives the whitener's output.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from ml4gw.nn.resnet import ResNet1D, ResNet2D
            >>> from ml4gw.nn.streaming import StreamRunner, StreamWhitener
            >>> from ml4gw.transforms import (
            ...     SingleQTransform, SpectralDensity, Whiten
            ... )
            >>> # Pass each kernel and its Q-transform to the model
            >>> class TimeQTransform(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.qtransform = SingleQTransform(
            ...             8, 2048, spectrogram_shape=(64, 128)
            ...         )
            ...     def forward(self, kernels):
            ...         return kernels, self.qtransform(kernels)
            >>> # A model with a time-domain and a Q-transform branch
            >>> class TimeQModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.time = ResNet1D(2, [2, 2], 1)
            ...         self.qscan = ResNet2D(2, [2, 2], 1)
            ...     def forward(self, X, X_q):
            ...         return self.time(X) + self.qscan(X_q)
            >>> whitener = StreamWhitener(
            ...     kernel_length=8, psd_length=64,
            ...     inference_sampling_rate=4, batch_size=16,
            ...     spectral_density=SpectralDensity(
            ...         2048, fftlength=2, average="median", fast=True
            ...     ),
            ...     transform=Whiten(2, 2048, highpass=32),
            ... )
            >>> runner = StreamRunner(
            ...     whitener=whitener,
            ...     model=TimeQModel(),
            ...     transform=TimeQTransform(),
            ... )
            >>> # (streams, channels, batch_size * stride_size)
            >>> update = torch.randn(2, 2, 16 * whitener.stride_size)
            >>> # y: (streams, batch_size, 1)
            >>> y = runner.step(update)
    """

    def __init__(
        self,
        whitener: StreamWhitener,
        model: torch.nn.Module,
        transform: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Set the buffer size from the whitener so the two agree
        self.buffer = StreamBuffer(
            whitener.input_size, whitener.batch_size * whitener.stride_size
        )
        self.whitener = whitener
        self.model = model
        self.transform = transform
        self.state: Tensor | None = None

    def reset(self) -> None:
        """Clear the state so that the next update starts new streams."""
        self.state = None

    @torch.no_grad()
    def step(
        self,
        update: Float[Tensor, "stream channel time"],
        psd_source: Int[Tensor, " stream"] | None = None,
    ) -> StreamOutput | tuple[StreamOutput, ...]:
        """
        Send a batch update with several streams through the model.

        Args:
            update:
                The new samples for this step, with shape
                ``(streams, channels, batch_size * stride)``.
            psd_source:
                ``streams`` indices. The ``i``th stream of ``update`` is
                whitened by the PSD of stream ``psd_source[i]``. If left
                as ``None``, each stream is whitened by its own PSD.

        Returns:
            Model outputs with shape
            ``(streams, batch_size, *output_shape)``. If the model
            returns a tuple, a tuple of outputs with these shapes.
        """
        # Append the new data to the streams' state. After a reset the
        # state is None and the buffer starts from zeros.
        x, self.state = self.buffer(update, self.state)
        kernels = self.whitener(x, psd_source)

        # Apply the optional transform, which is expected to take the
        # kernels and return an output of an appropriate shape for
        # self.model
        inputs = (
            self.transform(kernels) if self.transform is not None else kernels
        )

        # Run the model and regroup outputs by stream. Also handles
        # a model that takes a tuple of inputs, and/or returns a tuple
        # of multiple outputs.
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        y = self.model(*inputs)

        num_streams = x.shape[0]
        batch_size = self.whitener.batch_size
        if isinstance(y, tuple):
            return tuple(
                output.unflatten(0, (num_streams, batch_size)) for output in y
            )
        return y.unflatten(0, (num_streams, batch_size))

    def run(
        self,
        data: Float[Tensor, "stream channel time"],
        psd_source: Int[Tensor, " stream"] | None = None,
    ) -> StreamOutput | tuple[StreamOutput, ...]:
        """
        Run inference on every stream of a sequence at once.

        Args:
            data:
                Streams to run the model over, with shape
                ``(streams, channels, time)``.
            psd_source:
                ``streams`` indices. The ``i``th stream of ``update`` is
                whitened by the PSD of stream ``psd_source[i]``. If left
                as ``None``, each stream is whitened by its own PSD.

        Returns:
            Model outputs with shape
            ``(streams, layout.num_outputs, *output_shape)``, where
            ``layout`` is the :class:`StreamLayout` of the streams. If
            the model returns a tuple, a tuple of outputs with these
            shapes.
        """
        layout = StreamLayout(
            data.size(-1), self.whitener.batch_size, self.whitener.stride_size
        )
        self.reset()
        outputs = []
        for batch_idx in range(layout.num_batches):
            start, end, last = layout.batch_bounds(batch_idx)
            update = data[..., start:end]
            if last:
                update = torch.nn.functional.pad(update, (0, layout.num_pad))
            outputs.append(self.step(update, psd_source))

        # Concatenate each output's batches and trim the outputs from the
        # padded final batch. Handles multiple outputs.
        if isinstance(outputs[0], tuple):
            return tuple(
                torch.cat(y, dim=1)[:, : layout.num_outputs]
                for y in zip(*outputs, strict=True)
            )
        outputs = torch.cat(outputs, dim=1)
        return outputs[:, : layout.num_outputs]
