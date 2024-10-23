import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from torch import Tensor

from ml4gw.transforms import SplineInterpolate


class TestSplineInterpolate:
    def test_1d_interpolation(self):
        x_in = np.linspace(0, 10, 100)
        data = np.sin(x_in)
        # There are edge effects in the torch transform that
        # aren't present in scipy. Would be great to solve that,
        # but a workaround is to interpolate well within the
        # boundaries of the input coordinates
        x_out = np.linspace(1, 9, 200)

        scipy_spline = UnivariateSpline(x_in, data, k=3, s=0)
        expected = scipy_spline(x_out)

        torch_spline = SplineInterpolate(
            x_in=Tensor(x_in),
            x_out=Tensor(x_out),
            kx=3,
            sx=0,
        )
        actual = torch_spline(Tensor(data)).squeeze().numpy()

        # The "steady-state" ratio between the torch and scipy
        # interpolations is about 0.999, with some minor fluctuations.
        # Would be nice to know why the torch interpolation is
        # consistently smaller
        assert np.allclose(actual, expected, rtol=5e-3)

    def test_2d_interpolation(self):
        x_in = np.linspace(0, 10, 100)
        y_in = np.linspace(0, 5, 200)
        x_grid, y_grid = np.meshgrid(x_in, y_in)
        data = np.sin(x_grid) * np.cos(y_grid)
        x_out = np.linspace(1, 9, 200)
        y_out = np.linspace(1, 4, 50)

        scipy_spline = RectBivariateSpline(x_in, y_in, data.T, kx=3, ky=3, s=0)
        expected = scipy_spline(x_out, y_out).T

        torch_spline = SplineInterpolate(
            x_in=Tensor(x_in),
            x_out=Tensor(x_out),
            y_in=Tensor(y_in),
            y_out=Tensor(y_out),
            kx=3,
            ky=3,
            sx=0,
            sy=0,
        )
        actual = torch_spline(Tensor(data)).squeeze().numpy()

        # The "steady-state" ratio between the torch and scipy
        # interpolations is about 0.999, with some minor fluctuations.
        # Would be nice to know why the torch interpolation is
        # consistently smaller
        assert np.allclose(actual, expected, rtol=5e-3)
