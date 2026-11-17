import torch

from ml4gw.nn.streaming import StreamBuffer


def test_state_size_is_input_minus_update():
    buffer = StreamBuffer(
        input_size=1280,
        update_size=256,
    )
    assert buffer.state_size == 1024


def test_state_is_tail_of_output():
    buffer = StreamBuffer(
        input_size=1280,
        update_size=256,
    )

    update = torch.randn(4, 2, 256)

    x, state = buffer(update)

    torch.testing.assert_close(state, x[..., -1024:], rtol=0, atol=0)
    assert x.shape[-1] == 1280


def test_initial_state_is_on_update_device():
    buffer = StreamBuffer(
        input_size=1280,
        update_size=256,
    )

    update = torch.empty(4, 2, 256, device="meta")
    _, state = buffer(update)
    assert state.device == update.device


def test_state_carries_over_between_updates():
    buffer = StreamBuffer(
        input_size=1280,
        update_size=256,
    )

    first_update = torch.randn(4, 2, 256)
    second_update = torch.randn(4, 2, 256)

    _, state = buffer(first_update)
    x, state = buffer(second_update, state)

    torch.testing.assert_close(
        x[..., -2 * 256 :],
        torch.cat([first_update, second_update], dim=-1),
        rtol=0,
        atol=0,
    )
