import pytest
import torch

from ml4gw.transforms import ShiftedPearsonCorrelation


class TestShiftedPearsonCorrelation:
    @pytest.fixture
    def max_shift(self):
        return 5

    @pytest.fixture
    def transform(self, max_shift):
        return ShiftedPearsonCorrelation(max_shift)

    def check_in_range(self, corr):
        in_range = (-1 <= corr) & (corr <= 1)
        assert in_range.all().item()

    def test_x_ndim_3(self, transform, max_shift):
        expected_shape = (2 * max_shift + 1, 4, 2)

        x = torch.randn(2, 2048)

        # set up a y which is just a shifted
        # version of x at each batch index
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
        self.check_in_range(corr)

        # check that we get our maximum matches
        # at the expected indices
        idx = corr[:, :, 0].argmax(dim=0)
        expected_shifts = torch.arange(4) + max_shift - 2
        assert torch.equal(idx, expected_shifts)

        # and that those matches are nearly 1
        maxs = corr[idx, torch.arange(4), 0]
        expected_max = torch.ones(4)
        assert torch.allclose(maxs, expected_max, rtol=0.01)

        # do similar checks for 2dim y
        corr = transform(x, y[0])
        assert corr.shape == expected_shape
        self.check_in_range(corr)

        idx = corr[:, :, 0].argmax(dim=0)
        expected_shifts = torch.ones(4) * (max_shift - 2)
        assert torch.equal(idx, expected_shifts)

        maxs = corr[max_shift - 2, :, 0]
        assert torch.allclose(maxs, expected_max, rtol=0.01)

        # and finally for 1dim y. We've only been checking
        # against y's first channel, so this will just
        # look exactly like the last one.
        corr = transform(x, y[0, 0])
        assert corr.shape == expected_shape
        self.check_in_range(corr)

        idx = corr[:, :, 0].argmax(dim=0)
        assert torch.equal(idx, expected_shifts)

        maxs = corr[max_shift - 2, :, 0]
        assert torch.allclose(maxs, expected_max, rtol=0.01)
