import numpy as np
import pytest
import torch
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from torch import Tensor

from ml4gw.transforms import SplineInterpolate1D, SplineInterpolate2D


@pytest.mark.parametrize("x_out_len", [50, 100, 200])
def test_spline_1d_interpolation(x_out_len):
    x_min, x_max = 0, 10
    x_in = np.linspace(x_min, x_max, 100)
    data = np.sin(x_in)
    x_out = np.linspace(x_min, x_max, x_out_len)

    scipy_spline = UnivariateSpline(x_in, data, k=3, s=0)
    expected = scipy_spline(x_out)

    data_t = Tensor(data)
    x_in_t = Tensor(x_in)
    x_out_t = Tensor(x_out)

    torch_spline = SplineInterpolate1D(x_in=x_in_t, x_out=x_out_t, kx=3)
    actual = torch_spline(data_t).squeeze().numpy()
    assert np.allclose(actual, expected, rtol=1e-4)

    # Check that passing output grid behaves as expected
    actual = torch_spline(data_t, x_out_t).squeeze().numpy()
    assert np.allclose(actual, expected, rtol=1e-4)

    # Test data with height dimension
    height = 5
    data_batch = Tensor(data).repeat(5, 1)
    actual = torch_spline(data_batch).squeeze().numpy()
    for i in range(height):
        assert np.allclose(actual[i], expected, rtol=1e-4)


def test_spline_1d_errors():
    x_in = torch.arange(2)
    with pytest.raises(ValueError) as exc:
        SplineInterpolate1D(x_in=x_in)
    assert str(exc.value).startswith("Input x-coordinates must have")

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


@pytest.mark.parametrize("x_out_len", [50, 100, 200])
@pytest.mark.parametrize("y_out_len", [25, 200, 1000])
def test_spline_2d_interpolation(x_out_len, y_out_len):
    x_min, x_max = 0, 10
    y_min, y_max = 0, 5

    x_in = np.linspace(x_min, x_max, 100)
    y_in = np.linspace(y_min, y_max, 200)
    x_grid, y_grid = np.meshgrid(x_in, y_in)
    data = np.sin(x_grid) * np.cos(y_grid)

    x_out = np.linspace(x_min, x_max, x_out_len)
    y_out = np.linspace(y_min, y_max, y_out_len)

    scipy_spline = RectBivariateSpline(x_in, y_in, data.T, kx=3, ky=3, s=0)
    expected = scipy_spline(x_out, y_out).T

    data_t = Tensor(data)
    x_in_t, x_out_t = Tensor(x_in), Tensor(x_out)
    y_in_t, y_out_t = Tensor(y_in), Tensor(y_out)

    torch_spline = SplineInterpolate2D(
        x_in=x_in_t, x_out=x_out_t, y_in=y_in_t, y_out=y_out_t, kx=3, ky=3
    )
    actual = torch_spline(data_t).squeeze().numpy()
    assert np.allclose(actual, expected, rtol=1e-4)

    # Check that passing output grid behaves as expected
    actual = torch_spline(data_t, x_out_t, y_out_t).squeeze().numpy()
    assert np.allclose(actual, expected, rtol=1e-4)


def test_spline_2d_errors():
    x_in = torch.arange(2)
    y_in = torch.arange(2)
    with pytest.raises(ValueError) as exc:
        SplineInterpolate2D(x_in=x_in, y_in=y_in)
    assert str(exc.value).startswith("Input x-coordinates must have")

    with pytest.raises(ValueError) as exc:
        SplineInterpolate2D(x_in=torch.arange(10), y_in=y_in)
    assert str(exc.value).startswith("Input y-coordinates must have")

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
