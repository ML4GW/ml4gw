"""Benchmarks for SplineInterpolate1D and SplineInterpolate2D."""

import pytest
import torch

from ml4gw.transforms import SplineInterpolate1D, SplineInterpolate2D


@pytest.fixture(params=[128, 512], ids=lambda x: f"num_points_{x}")
def num_points(request):
    return request.param


def test_spline_1d_forward(benchmark, batch_size, num_points, device):
    x_in = torch.linspace(0, 1, num_points, device=device)
    x_out = torch.linspace(0, 1, num_points * 2, device=device)
    interpolator = SplineInterpolate1D(
        x_in=x_in.cpu(), x_out=x_out.cpu(), kx=3
    ).to(device)
    data = torch.randn(batch_size, 1, 1, num_points, device=device)
    benchmark(interpolator, data)


def test_spline_2d_forward(benchmark, batch_size, num_points, device):
    x_in = torch.linspace(0, 1, num_points)
    y_in = torch.linspace(0, 1, num_points)
    x_out = torch.linspace(0, 1, num_points * 2)
    y_out = torch.linspace(0, 1, num_points * 2)
    interpolator = SplineInterpolate2D(
        x_in=x_in, y_in=y_in, x_out=x_out, y_out=y_out, kx=3, ky=3
    ).to(device)
    data = torch.randn(batch_size, 1, num_points, num_points, device=device)
    benchmark(interpolator, data)
