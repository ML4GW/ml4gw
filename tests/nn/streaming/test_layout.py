from ml4gw.nn.streaming import StreamLayout


def test_even_split_has_no_padding_and_full_final_batch():
    layout = StreamLayout(
        size=2048,
        batch_size=128,
        stride=4,
    )

    assert layout.num_batches == 4
    assert layout.num_pad == 0
    assert layout.num_outputs == 512

    start, end, last = layout.batch_bounds(3)
    assert start == 1536
    assert end == 2048
    assert last


def test_uneven_split_drops_outputs_from_padding():
    layout = StreamLayout(
        size=2048,
        batch_size=128,
        stride=5,
    )

    assert layout.num_batches == 4
    assert layout.num_pad == 512
    # The 512 samples in the pad covers 102 strides
    assert layout.num_outputs == 410

    start, end, last = layout.batch_bounds(3)
    # Make sure that we don't go out of bounds with uneven batches
    assert start == 1920
    assert end == 2048
    assert last
