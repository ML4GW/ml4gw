import pytest
import torch


def test_sizes(make_whitener):
    stream_whitener = make_whitener()

    assert stream_whitener.stride_size == 16
    # 3 strides of 16 + 2048 kernel size + 4096 filter size
    assert stream_whitener.size == 6192
    # size + 8 * 2048 psd size
    assert stream_whitener.input_size == 22_576


def test_sample_rate_mismatch_raises(make_whitener):
    with pytest.raises(ValueError, match="doesn't match whitening"):
        make_whitener(psd_sample_rate=4096)


def test_asd_ignores_data_segment(make_whitener):
    stream_whitener = make_whitener()
    sample = torch.randn(2, 2, stream_whitener.input_size)

    size = stream_whitener.size
    sample_modified = sample.clone()
    sample_modified[..., -size:] = torch.randn(2, 2, size)

    _, asd = stream_whitener.whiten(sample)
    _, asd_modified = stream_whitener.whiten(sample_modified)

    # The ASD should not change based on the data being whitened
    torch.testing.assert_close(asd, asd_modified)


def test_psd_source_swaps_asds(make_whitener):
    stream_whitener = make_whitener()
    sample = torch.randn(2, 2, stream_whitener.input_size)
    psd_source = torch.as_tensor([1, 0])

    _, asd = stream_whitener.whiten(sample)
    _, asd_swapped = stream_whitener.whiten(sample, psd_source)

    # psd_source should swap the ASDs relative to leaving it None
    torch.testing.assert_close(asd[0], asd_swapped[1])
    torch.testing.assert_close(asd[1], asd_swapped[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_matches_input_dtype(make_whitener, dtype):
    stream_whitener = make_whitener()
    sample = torch.randn(2, 2, stream_whitener.input_size, dtype=dtype)

    whitened, _ = stream_whitener.whiten(sample)
    assert whitened.dtype == sample.dtype


def test_unfolded_kernels_ordered_by_stream(make_whitener):
    stream_whitener = make_whitener(
        kernel_length=3 / 2048,  # 3 samples
        inference_sampling_rate=1024,  # stride of 2 samples
        batch_size=2,
    )
    # 2 streams, 1 channel, (batch_size - 1) * stride + kernel = 5 samples
    whitened = torch.arange(10.0).reshape(2, 1, 5)

    kernels = stream_whitener.unfold(whitened)

    expected = torch.tensor(
        [
            [[0.0, 1.0, 2.0]],
            [[2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0]],
            [[7.0, 8.0, 9.0]],
        ]
    )
    torch.testing.assert_close(kernels, expected, rtol=0, atol=0)


def test_forward_returns_batch_size_kernels_per_stream(make_whitener):
    stream_whitener = make_whitener()
    sample = torch.randn(2, 2, stream_whitener.input_size)

    kernels = stream_whitener(sample)

    # Should have shape (batch_size * streams, num_channels, kernel_size)
    assert kernels.shape == (8, 2, 2048)
