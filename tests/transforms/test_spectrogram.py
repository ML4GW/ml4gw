import pytest
import torch
from torchaudio.transforms import Spectrogram

from ml4gw.transforms import MultiResolutionSpectrogram


@pytest.fixture(params=[2, 4, 5])
def kernel_length(request):
    return request.param


@pytest.fixture(params=[128, 256])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[1, 10])
def batch_size(request):
    return request.param


@pytest.fixture(params=[1, 3])
def num_channels(request):
    return request.param


@pytest.fixture(params=[[64], [64, 128], [64, 128, 256]])
def n_ffts(request):
    return request.param


@pytest.fixture(params=[[50]])
def win_lengths(request):
    return request.param


@pytest.fixture(params=[[2]])
def powers(request):
    return request.param


def test_num_spectrogram_warning():
    with pytest.warns(RuntimeWarning):
        MultiResolutionSpectrogram(8, 2048, n_fft=[32, 64, 128, 256])


def test_normalization_error():
    with pytest.raises(ValueError):
        MultiResolutionSpectrogram(8, 2048, normalized=[False])


def test_arg_length_error():
    with pytest.raises(ValueError):
        MultiResolutionSpectrogram(8, 2048, n_fft=[32, 64], power=[1, 1, 1])


def test_kernel_size_error():
    with pytest.raises(ValueError):
        spectrogram = MultiResolutionSpectrogram(8, 2048)
        spectrogram(torch.ones([4, 3, 8 * 2048 + 1]))


def test_multi_resolution_spectrogram(
    kernel_length,
    sample_rate,
    batch_size,
    num_channels,
    n_ffts,
    win_lengths,
    powers,
):

    # Creating a MRS without any spectrogram arguments should
    # just create a single default torchaudio histogram with
    # `normalized = True`
    spectrogram = MultiResolutionSpectrogram(kernel_length, sample_rate)

    X = torch.ones([batch_size, num_channels, kernel_length * sample_rate])
    y = spectrogram(X)
    expected_y = Spectrogram(normalized=True)(X)

    assert (y == expected_y).all()

    # Check that all the indexing we're doing is working by
    # performing a more explicit version of the algorithm
    spectrogram = MultiResolutionSpectrogram(
        kernel_length,
        sample_rate,
        n_fft=n_ffts,
        win_length=win_lengths,
        power=powers,
    )
    y = spectrogram(X)
    kwargs = spectrogram.kwargs
    ta_spectrograms = [Spectrogram(**k)(X[0, 0]) for k in kwargs]
    t_dim = max([spec.shape[-1] for spec in ta_spectrograms])
    f_dim = max([spec.shape[-2] for spec in ta_spectrograms])
    expected_y = torch.zeros([f_dim, t_dim])

    for i in range(t_dim):
        for j in range(f_dim):
            max_value = 0
            for spec in ta_spectrograms:
                t_idx = int(i / t_dim * spec.shape[-1])
                f_idx = int(j / f_dim * spec.shape[-2])
                max_value = max(max_value, spec[f_idx, t_idx])
            expected_y[j, i] += max_value

    assert torch.allclose(y[0, 0], expected_y, rtol=1e-6)
