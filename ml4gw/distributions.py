"""
Module containing callables classes for generating samples
from specified distributions. Each callable should map from
an integer `N` to a 1D torch `Tensor` containing `N` samples
from the corresponding distribution.
"""

import math
from typing import Optional

import torch
import torch.distributions as dist
from jaxtyping import Float
from torch import Tensor

from ml4gw.constants import C

_PLANCK18_H0 = 67.66  # Hubble constant in km/s/Mpc
_PLANCK18_OMEGA_M = 0.30966  # Matter density parameter


class Cosine(dist.Distribution):
    """
    Cosine distribution based on
    ``torch.distributions.TransformedDistribution``.
    """

    arg_constraints = {}

    def __init__(
        self,
        low: float = -math.pi / 2,
        high: float = math.pi / 2,
        validate_args=None,
    ):
        batch_shape = torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.norm = 1 / (torch.sin(self.high) - torch.sin(self.low))

    def rsample(self, sample_shape: torch.Size = None) -> Tensor:
        sample_shape = sample_shape or torch.Size()
        u = torch.rand(sample_shape, device=self.low.device)
        return torch.arcsin(u / self.norm + torch.sin(self.low))

    def log_prob(self, value: float) -> Float[Tensor, ""]:
        value = torch.as_tensor(value)
        inside_range = (value >= self.low) & (value <= self.high)
        return (value.cos() * inside_range).log()


class Sine(dist.TransformedDistribution):
    """
    Sine distribution based on
    ``torch.distributions.TransformedDistribution``.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = math.pi,
        validate_args=None,
    ):
        low = torch.as_tensor(low)
        high = torch.as_tensor(high)
        base_dist = Cosine(
            low - torch.pi / 2, high - torch.pi / 2, validate_args
        )

        super().__init__(
            base_dist,
            [
                dist.AffineTransform(
                    loc=torch.pi / 2,
                    scale=1,
                )
            ],
            validate_args=validate_args,
        )


class LogUniform(dist.TransformedDistribution):
    """
    Sample from a log uniform distribution
    """

    def __init__(self, low: float, high: float, validate_args=None):
        base_dist = dist.Uniform(
            torch.as_tensor(low).log(),
            torch.as_tensor(high).log(),
            validate_args,
        )
        super().__init__(
            base_dist,
            [dist.ExpTransform()],
            validate_args=validate_args,
        )


class LogNormal(dist.LogNormal):
    def __init__(
        self,
        mean: float,
        std: float,
        low: Optional[float] = None,
        validate_args=None,
    ):
        self.low = low
        super().__init__(loc=mean, scale=std, validate_args=validate_args)

    def support(self):
        if self.low is not None:
            return dist.constraints.greater_than(self.low)


class PowerLaw(dist.TransformedDistribution):
    """
    Sample from a power law distribution,
    .. math::
        p(x) \approx x^{\alpha}.

    Index alpha cannot be 0, since it is equivalent to a Uniform distribution.
    This could be used, for example, as a universal distribution of
    signal-to-noise ratios (SNRs) from uniformly volume distributed
    sources
    .. math::

       p(\rho) = 3*\rho_0^3 / \rho^4

    where :math:`\rho_0` is a representative minimum SNR
    considered for detection. See, for example,
    `Schutz (2011) <https://arxiv.org/abs/1102.5421>`_.
    Or, for example, ``index=2`` for uniform in Euclidean volume.
    """

    support = dist.constraints.nonnegative

    def __init__(
        self, minimum: float, maximum: float, index: int, validate_args=None
    ):
        if index == 0:
            raise RuntimeError("Index of 0 is the same as Uniform")
        elif index == -1:
            base_min = torch.as_tensor(minimum).log()
            base_max = torch.as_tensor(maximum).log()
            transforms = [dist.ExpTransform()]
        else:
            index_plus = index + 1
            base_min = minimum**index_plus / index_plus
            base_max = maximum**index_plus / index_plus
            transforms = [
                dist.AffineTransform(loc=0, scale=index_plus),
                dist.PowerTransform(1 / index_plus),
            ]
        base_dist = dist.Uniform(base_min, base_max, validate_args=False)
        super().__init__(
            base_dist,
            transforms,
            validate_args=validate_args,
        )


class DeltaFunction(dist.Distribution):
    arg_constraints = {}

    def __init__(
        self,
        peak: float = 0.0,
        validate_args=None,
    ):
        batch_shape = torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)
        self.peak = torch.as_tensor(peak)

    def rsample(self, sample_shape: torch.Size = None) -> Tensor:
        sample_shape = sample_shape or torch.Size()
        return self.peak * torch.ones(
            sample_shape, device=self.peak.device, dtype=torch.float32
        )


class UniformComovingVolume(dist.Distribution):
    """
    Sample either redshift, comoving distance, or luminosity distance
    such that they are uniform in comoving volume, assuming a flat
    lambda-CDM cosmology. Default H0 and Omega_M values match
    astropy.cosmology.Planck18

    Args:
        minimum: Minimum distance in the specified distance type
        maximum: Maximum distance in the specified distance type
        distance_type:
            Type of distance to sample from. Can be 'redshift',
            'comoving_distance', or 'luminosity_distance'
        source_frame_time:
            Whether to additionally make the distribution uniform
            in source frame time
        h0: Hubble constant in km/s/Mpc
        omega_m: Matter density parameter
        z_max: Maximum redshift for the grid
        grid_size: Number of points in the grid for interpolation
        validate_args: Whether to validate arguments
    """

    arg_constraints = {}
    support = dist.constraints.nonnegative

    def __init__(
        self,
        minimum: float,
        maximum: float,
        distance_type: str = "redshift",
        source_frame_time: bool = False,
        h0: float = 67.66,
        omega_m: float = 0.30966,
        z_grid_max: float = 5,
        grid_size: int = 10000,
        validate_args: bool = None,
    ):
        super().__init__(validate_args=validate_args)

        if distance_type not in [
            "redshift",
            "comoving_distance",
            "luminosity_distance",
        ]:
            raise ValueError(
                "Distance type must be 'redshift', 'comoving_distance', "
                f"or 'luminosity_distance'; got {distance_type}"
            )

        self.distance_type = distance_type
        self.grid_size = grid_size

        z_grid = torch.linspace(0, z_grid_max, grid_size)

        omega_l = 1 - omega_m
        h = h0 * torch.sqrt(omega_m * (1 + z_grid) ** 3 + omega_l)

        dz = z_grid[1] - z_grid[0]

        # C is specfied in m/s, h0 in km/s/Mpc, so divide by 1000 to convert
        comoving_dist_grid = torch.cumsum((C / h) * dz, dim=0) / 1000
        # Compute luminosity distance from comoving distance
        luminosity_dist_grid = comoving_dist_grid * (1 + z_grid)

        if distance_type == "comoving_distance":
            self.minimum, self.maximum = Tensor([minimum, maximum])
            z_min, z_max = self._linear_interp_1d(
                comoving_dist_grid, z_grid, Tensor([minimum, maximum])
            )
        elif distance_type == "redshift":
            self.minimum, self.maximum = self._linear_interp_1d(
                z_grid, comoving_dist_grid, Tensor([minimum, maximum])
            )
            z_min, z_max = Tensor([minimum, maximum])
        else:
            self.minimum, self.maximum = self._linear_interp_1d(
                luminosity_dist_grid,
                comoving_dist_grid,
                Tensor([minimum, maximum]),
            )
            z_min, z_max = self._linear_interp_1d(
                luminosity_dist_grid, z_grid, Tensor([minimum, maximum])
            )

        if self.maximum > comoving_dist_grid[-1]:
            raise ValueError(
                f"Maximum comoving distance {self.maximum} "
                f"exceeds given z_max {z_grid_max}."
            )

        # Restrict grids based on the minimum and maximum values
        mask = (z_grid >= z_min) & (z_grid <= z_max)
        z_grid = z_grid[mask]
        comoving_dist_grid = comoving_dist_grid[mask]
        luminosity_dist_grid = luminosity_dist_grid[mask]
        h = h[mask]

        # Compute the pdf and log_pdf from the comoving distance grid
        dV_dz = comoving_dist_grid**2 / h
        if source_frame_time:
            dV_dz /= 1 + z_grid
        pdf_grid = dV_dz / torch.sum(dV_dz * dz)

        log_pdf = torch.log(pdf_grid)

        # Compute change of variables factors for computing log_prob
        dcomoving_dist_dz = torch.gradient(comoving_dist_grid, spacing=dz)[0]
        dluminosity_dist_dz = torch.gradient(luminosity_dist_grid, spacing=dz)[
            0
        ]

        self.z_grid = z_grid
        self.dz = dz
        self.comoving_dist_grid = comoving_dist_grid
        self.luminosity_dist_grid = luminosity_dist_grid
        self.log_pdf = log_pdf
        self.dz_dcomoving_dist = 1 / dcomoving_dist_dz
        self.dz_dluminosity_dist = 1 / dluminosity_dist_dz

    def _linear_interp_1d(self, x_grid, y_grid, x_query):
        idx = torch.bucketize(x_query, x_grid, right=True)
        idx = idx.clamp(min=1, max=len(x_grid) - 1)

        x0 = x_grid[idx - 1]
        x1 = x_grid[idx]
        y0 = y_grid[idx - 1]
        y1 = y_grid[idx]

        # Factor of epsilon for stability
        t = (x_query - x0) / (x1 - x0 + 1e-20)
        return y0 + t * (y1 - y0)

    def rsample(self, sample_shape: torch.Size = None) -> Tensor:
        sample_shape = sample_shape or torch.Size()
        u = torch.rand(sample_shape)

        pdf = self.log_pdf.exp()
        cdf = torch.cumsum(pdf * self.dz, dim=0)
        z = self._linear_interp_1d(cdf, self.z_grid, u)

        if self.distance_type == "redshift":
            return z
        elif self.distance_type == "comoving_distance":
            return self._linear_interp_1d(
                self.z_grid, self.comoving_dist_grid, z
            )
        else:
            return self._linear_interp_1d(
                self.z_grid, self.luminosity_dist_grid, z
            )

    def log_prob(self, value: Tensor) -> Tensor:
        if self.distance_type == "redshift":
            return self._linear_interp_1d(self.z_grid, self.log_pdf, value)
        elif self.distance_type == "comoving_distance":
            z = self._linear_interp_1d(
                self.comoving_dist_grid, self.z_grid, value
            )
            log_prob = self._linear_interp_1d(self.z_grid, self.log_pdf, z)
            dz_dcomoving_dist = self._linear_interp_1d(
                self.comoving_dist_grid, self.dz_dcomoving_dist, value
            )
            return log_prob + torch.log(dz_dcomoving_dist)
        else:
            z = self._linear_interp_1d(
                self.luminosity_dist_grid, self.z_grid, value
            )
            log_prob = self._linear_interp_1d(self.z_grid, self.log_pdf, z)
            dz_dluminosity_dist = self._linear_interp_1d(
                self.luminosity_dist_grid, self.dz_dluminosity_dist, value
            )
            return log_prob + torch.log(dz_dluminosity_dist)
