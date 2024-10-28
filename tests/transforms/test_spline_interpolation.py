import numpy as np
import pytest
import torch
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from torch import Tensor

from ml4gw.transforms import SplineInterpolate


class TestSplineInterpolate:
    @pytest.fixture(params=[50, 100, 200])
    def x_out_len(self, request):
        return request.param

    @pytest.fixture(params=[25, 200, 1000])
    def y_out_len(self, request):
        return request.param

    def test_1d_interpolation(self, x_out_len):
        x_in = np.linspace(0, 10, 100)
        data = np.sin(x_in)
        # There are edge effects in the torch transform that
        # aren't present in scipy. Would be great to solve that,
        # but a workaround is to interpolate well within the
        # boundaries of the input coordinates. Unfortunately,
        # what specifically that means depends on the size of
        # the input array.
        pad = len(x_in) // 10
        x_out = np.linspace(x_in[pad], x_in[-pad], x_out_len)

        scipy_spline = UnivariateSpline(x_in, data, k=3, s=0)
        expected = scipy_spline(x_out)

        torch_spline = SplineInterpolate(
            x_in=Tensor(x_in),
            x_out=Tensor(x_out),
            kx=3,
        )
        actual = torch_spline(Tensor(data)).squeeze().numpy()

        # The "steady-state" ratio between the torch and scipy
        # interpolations is about 0.9990, with some minor fluctuations.
        # Would be nice to know why the torch interpolation is
        # consistently smaller
        assert np.allclose(actual, expected, rtol=5e-3)

    def test_2d_interpolation(self, x_out_len, y_out_len):
        x_in = np.linspace(0, 10, 100)
        y_in = np.linspace(0, 5, 200)
        x_grid, y_grid = np.meshgrid(x_in, y_in)
        data = np.sin(x_grid) * np.cos(y_grid)

        pad = len(x_in) // 10
        x_out = np.linspace(x_in[pad], x_in[-pad], x_out_len)
        pad = len(y_in) // 10
        y_out = np.linspace(y_in[pad], y_in[-pad], y_out_len)

        scipy_spline = RectBivariateSpline(x_in, y_in, data.T, kx=3, ky=3, s=0)
        expected = scipy_spline(x_out, y_out).T

        torch_spline = SplineInterpolate(
            x_in=Tensor(x_in),
            x_out=Tensor(x_out),
            y_in=Tensor(y_in),
            y_out=Tensor(y_out),
            kx=3,
            ky=3,
        )
        actual = torch_spline(Tensor(data)).squeeze().numpy()

        # The "steady-state" ratio between the torch and scipy
        # interpolations is about 0.999, with some minor fluctuations.
        # Would be nice to know why the torch interpolation is
        # consistently smaller
        assert np.allclose(actual, expected, rtol=5e-3)

    def test_errors(self):
        x_in = torch.arange(10)
        x_out = x_in
        torch_spline = SplineInterpolate(x_in)
        data = torch.randn(len(x_in))
        with pytest.raises(ValueError) as exc:
            torch_spline(data)
        assert str(exc.value).startswith("Output x-coordinates were not")

        data = torch.randn((1, 2, 3, 4, 5))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("Input data has more than 4")

        y_in = torch.arange(10)
        torch_spline = SplineInterpolate(x_in=x_in, y_in=y_in)
        data = torch.randn(len(x_in))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("An input y-coordinate array")

        data = torch.randn((len(y_in) - 1, len(x_in) - 1))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("The spatial dimensions of the data")
