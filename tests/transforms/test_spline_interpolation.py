import numpy as np
import pytest
import torch
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from torch import Tensor

from ml4gw.transforms import SplineInterpolate1D, SplineInterpolate2D


class TestSplineInterpolate1D:
    @pytest.fixture(params=[50, 100, 200])
    def x_out_len(self, request):
        return request.param

    def test_interpolation(self, x_out_len):
        x_min = 0
        x_max = 10

        x_in = np.linspace(x_min, x_max, 100)
        data = np.sin(x_in)
        x_out = np.linspace(x_min, x_max, x_out_len)

        scipy_spline = UnivariateSpline(x_in, data, k=3, s=0)
        expected = scipy_spline(x_out)

        data = Tensor(data)
        x_in = Tensor(x_in)
        x_out = Tensor(x_out)

        torch_spline = SplineInterpolate1D(
            x_in=x_in,
            x_out=x_out,
            kx=3,
        )
        actual = torch_spline(data).squeeze().numpy()

        assert np.allclose(actual, expected, rtol=1e-4)

        # Check that passing output grid behaves as expected
        actual = torch_spline(data, x_out).squeeze().numpy()
        assert np.allclose(actual, expected, rtol=1e-4)

        # Test data with height dimension
        height = 5
        data = Tensor(data).repeat(5, 1)
        actual = torch_spline(data).squeeze().numpy()
        for i in range(height):
            assert np.allclose(actual[i], expected, rtol=1e-4)

    def test_errors(self):
        x_in = torch.arange(10)
        x_out = x_in
        torch_spline = SplineInterpolate1D(x_in)
        data = torch.randn(len(x_in))
        with pytest.raises(ValueError) as exc:
            torch_spline(data)
        assert str(exc.value).startswith("Output x-coordinates were not")

        data = torch.randn((1, 2, 3, 4, 5))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("Input data has more than 4")

        data = torch.randn(len(x_in) - 1)
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("The spatial dimensions of the data")


class TestSplineInterpolate2D:
    @pytest.fixture(params=[50, 100, 200])
    def x_out_len(self, request):
        return request.param

    @pytest.fixture(params=[25, 200, 1000])
    def y_out_len(self, request):
        return request.param

    def test_interpolation(self, x_out_len, y_out_len):
        x_min = 0
        x_max = 10
        y_min = 0
        y_max = 5

        x_in = np.linspace(x_min, x_max, 100)
        y_in = np.linspace(y_min, y_max, 200)
        x_grid, y_grid = np.meshgrid(x_in, y_in)
        data = np.sin(x_grid) * np.cos(y_grid)

        x_out = np.linspace(x_min, x_max, x_out_len)
        y_out = np.linspace(y_min, y_max, y_out_len)

        scipy_spline = RectBivariateSpline(x_in, y_in, data.T, kx=3, ky=3, s=0)
        expected = scipy_spline(x_out, y_out).T

        data = Tensor(data)
        x_in = Tensor(x_in)
        x_out = Tensor(x_out)
        y_in = Tensor(y_in)
        y_out = Tensor(y_out)

        torch_spline = SplineInterpolate2D(
            x_in=x_in,
            x_out=x_out,
            y_in=y_in,
            y_out=y_out,
            kx=3,
            ky=3,
        )
        actual = torch_spline(data).squeeze().numpy()

        assert np.allclose(actual, expected, rtol=1e-4)

        # Check that passing output grid behaves as expected
        actual = torch_spline(data, x_out, y_out).squeeze().numpy()
        assert np.allclose(actual, expected, rtol=1e-4)

    def test_errors(self):
        x_in = torch.arange(10)
        x_out = x_in
        y_in = torch.arange(10)
        y_out = y_in
        torch_spline = SplineInterpolate2D(x_in=x_in, y_in=y_in)
        data = torch.randn(len(x_in))
        with pytest.raises(ValueError) as exc:
            torch_spline(data)
        assert str(exc.value).startswith("Output x-coordinates were not")

        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out)
        assert str(exc.value).startswith("Output y-coordinates were not")

        data = torch.randn((1, 2, 3, 4, 5))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out, y_out=y_out)
        assert str(exc.value).startswith("Input data has more than 4")

        data = torch.randn(10)
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out, y_out=y_out)
        assert str(exc.value).startswith("Input data has fewer than 2")

        data = torch.randn((len(y_in) - 1, len(x_in) - 1))
        with pytest.raises(ValueError) as exc:
            torch_spline(data, x_out=x_out, y_out=y_out)
        assert str(exc.value).startswith("The spatial dimensions of the data")
