import torch
from jaxtyping import Float
from torch import Tensor


class StreamBuffer(torch.nn.Module):
    """
    Append a streaming update to the trailing samples of the previous one.

    Concatenates each new streaming update with the state from the
    previous update, and extracts the appropriately sized window as the
    state for the next iteration. The state is passed in and returned
    rather than stored, so the caller keeps it between updates. Useful
    for processing continuous data streams in chunks.

    Args:
        state_size:
            Number of trailing samples kept as state between updates.
            For streams passed to a :class:`StreamWhitener`, this is
            its ``state_size``.

    Example:
        .. code-block:: python

            >>> import torch
            >>> from ml4gw.nn.streaming import StreamBuffer
            >>> buffer = StreamBuffer(state_size=1024)
            >>> # (streams, channels, samples)
            >>> update = torch.randn(4, 2, 256)
            >>> state = buffer.get_initial_state(4, 2)
            >>> x, new_state = buffer(update, state)
    """

    def __init__(self, state_size: int) -> None:
        super().__init__()
        self.state_size = state_size

    def get_initial_state(
        self,
        num_streams: int,
        num_channels: int,
        device: torch.device | None = None,
    ) -> Float[Tensor, "stream channel time"]:
        return torch.zeros(
            (num_streams, num_channels, self.state_size), device=device
        )

    def forward(
        self,
        update: Float[Tensor, "stream channel time1"],
        state: Float[Tensor, "stream channel time2"] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Concatenate new update with state and extract sliding window.

        Args:
            update:
                New data chunk to append, with shape
                ``(streams, channels, samples)``.
            state:
                Previous sliding window state, with shape
                ``(streams, channels, state_size)``. If left as
                ``None``, zeros on the update's device are used,
                as for the first update of new streams.

        Returns:
            The concatenated full data, and the updated sliding
            window state to pass to the next call.
        """
        if state is None:
            state = self.get_initial_state(
                update.size(0), update.size(1), update.device
            )
        # Concatenate new data with previous state
        x = torch.cat([state, update], dim=-1)
        # Extract the sliding window from the end of concatenated data
        state = x[:, :, -self.state_size :]
        return x, state
