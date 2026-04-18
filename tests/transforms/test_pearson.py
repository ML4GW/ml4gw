import pytest
import torch

from ml4gw.transforms import ShiftedPearsonCorrelation


@pytest.fixture
def max_shift():
    return 5


@pytest.fixture
def transform(max_shift):
    return ShiftedPearsonCorrelation(max_shift)


def test_shifted_pearson_correlation(transform, max_shift):
    expected_shape = (2 * max_shift + 1, 4, 2)

    x = torch.randn(2, 2048)

    # set up a y which is just a shifted version of x at each batch index
    y = torch.zeros((4, 2, 2048))
    for i in range(4):
        j = i - 2
        if j < 0:
            y[i, :, -j:] = x[:, :j]
        elif j > 0:
            y[i, :, :-j] = x[:, j:]
        else:
            y[i] = x

    # make all batch elements of x the same
    x = x.view(1, 2, -1).repeat(4, 1, 1)
    corr = transform(x, y)

    # first check that we have the expected shape
    assert corr.shape == expected_shape

    # check that all our values are in the expected range
    assert ((-1 <= corr) & (corr <= 1)).all().item()

    # check that we get our maximum matches at the expected indices
    idx = corr[:, :, 0].argmax(dim=0)
    expected_shifts = torch.arange(4) + max_shift - 2
    assert torch.equal(idx, expected_shifts)

    # and that those matches are nearly 1
    maxs = corr[idx, torch.arange(4), 0]
    assert torch.allclose(maxs, torch.ones(4), rtol=0.01)

    # do similar checks for 2dim y
    corr = transform(x, y[0])
    assert corr.shape == expected_shape
    assert ((-1 <= corr) & (corr <= 1)).all().item()

    idx = corr[:, :, 0].argmax(dim=0)
    expected_shifts = torch.ones(4) * (max_shift - 2)
    assert torch.equal(idx, expected_shifts)

    maxs = corr[max_shift - 2, :, 0]
    assert torch.allclose(maxs, torch.ones(4), rtol=0.01)

    # and finally for 1dim y; we've only been checking against y's first
    # channel, so this will look exactly like the last case
    corr = transform(x, y[0, 0])
    assert corr.shape == expected_shape
    assert ((-1 <= corr) & (corr <= 1)).all().item()

    idx = corr[:, :, 0].argmax(dim=0)
    assert torch.equal(idx, expected_shifts)

    maxs = corr[max_shift - 2, :, 0]
    assert torch.allclose(maxs, torch.ones(4), rtol=0.01)


def test_shifted_pearson_shape_errors(transform):
    x = torch.randn(2, 2, 2048)

    with pytest.raises(ValueError, match="up to 3 dimensions"):
        transform(x[None], x[None])

    with pytest.raises(ValueError, match="more dimensions"):
        transform(x[0], x)

    with pytest.raises(ValueError, match="same size along last"):
        transform(x, x[..., :-1])
