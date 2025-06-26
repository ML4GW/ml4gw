"""
Module containing callables classes for generating samples
from specified distributions. Each callable should map from
an integer `N` to a 1D torch `Tensor` containing `N` samples
from the corresponding distribution.
"""

import math
from typing import Callable, Optional

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
        h0: float = _PLANCK18_H0,
        omega_m: float = _PLANCK18_OMEGA_M,
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

        self.minimum = minimum
        self.maximum = maximum
        self.distance_type = distance_type
        self.grid_size = grid_size
        self.z_grid_max = z_grid_max
        self.h0 = h0
        self.omega_m = omega_m

        # Compute redshift range based on the given min and max distances
        z_min, z_max = self._get_z_bounds()
        if z_max > z_grid_max:
            raise ValueError(
                f"Maximum {distance_type} {maximum} "
                f"exceeds given z_max {z_grid_max}."
            )

        # Restrict distance grids to the specified redshift range
        mask = (self.z_grid >= z_min) & (self.z_grid <= z_max)
        self.distance_grid = self.distance_grid[mask]
        self.z_grid = self.z_grid[mask]
        self.comoving_dist_grid = self.comoving_dist_grid[mask]
        self.luminosity_dist_grid = self.luminosity_dist_grid[mask]
        # Compute probability arrays from those grids
        self._generate_probability_grids()

    def _hubble_function(self):
        """
        Compute H(z) assuming a flat lambda-CDM cosmology.
        """
        omega_l = 1 - self.omega_m
        return self.h0 * torch.sqrt(
            self.omega_m * (1 + self.z_grid) ** 3 + omega_l
        )

    def _get_z_bounds(self):
        """
        Compute the bounds on redshift based on the given minimum and maximum
        distances, using the specified distance type.
        """
        self._generate_distance_grids()
        bounds = torch.tensor([self.minimum, self.maximum])
        z_min, z_max = self._linear_interp_1d(
            self.distance_grid, self.z_grid, bounds
        )

        return z_min, z_max

    def _generate_distance_grids(self):
        """
        Generate distance grids based on the specified redshift range.
        """
        self.z_grid = torch.linspace(0, self.z_grid_max, self.grid_size)
        self.dz = self.z_grid[1] - self.z_grid[0]
        # C is specfied in m/s, h0 in km/s/Mpc, so divide by 1000 to convert
        comoving_dist_grid = (
            torch.cumulative_trapezoid(
                (C / self._hubble_function()), self.z_grid
            )
            / 1000
        )
        zero_prefix = torch.zeros(1, dtype=comoving_dist_grid.dtype)
        self.comoving_dist_grid = torch.cat([zero_prefix, comoving_dist_grid])
        self.luminosity_dist_grid = self.comoving_dist_grid * (1 + self.z_grid)

        if self.distance_type == "redshift":
            self.distance_grid = self.z_grid
        elif self.distance_type == "comoving_distance":
            self.distance_grid = self.comoving_dist_grid
        else:  # luminosity_distance
            self.distance_grid = self.luminosity_dist_grid

    def _p_of_distance(self):
        """
        Compute the unnormalized probability as a function of distance
        """
        dV_dz = self.comoving_dist_grid**2 / self._hubble_function()
        # This is a tensor of ones if the distance type is redshift
        jacobian = torch.gradient(self.distance_grid, spacing=self.dz)[0]
        return dV_dz / jacobian

    def _generate_probability_grids(self):
        """
        Compute the pdf, cdf, and log pdf based on the
        comoving volume differential and distance grid.
        """
        p_of_distance = self._p_of_distance()
        self.pdf = p_of_distance / torch.trapz(
            p_of_distance, self.distance_grid
        )
        cdf = torch.cumulative_trapezoid(self.pdf, self.distance_grid)
        zero_prefix = torch.zeros(1, dtype=cdf.dtype)
        self.cdf = torch.cat([zero_prefix, cdf])
        self.log_pdf = torch.log(self.pdf)

    def _linear_interp_1d(self, x_grid, y_grid, x_query):
        idx = torch.bucketize(x_query, x_grid, right=True)
        idx = idx.clamp(min=1, max=len(x_grid) - 1)

        x0 = x_grid[idx - 1]
        x1 = x_grid[idx]
        y0 = y_grid[idx - 1]
        y1 = y_grid[idx]

        t = (x_query - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    def rsample(self, sample_shape: torch.Size = None) -> Tensor:
        sample_shape = sample_shape or torch.Size()
        u = torch.rand(sample_shape)
        return self._linear_interp_1d(self.cdf, self.distance_grid, u)

    def log_prob(self, value: Tensor) -> Tensor:
        log_prob = self._linear_interp_1d(
            self.distance_grid, self.log_pdf, value
        )
        inside_range = (value >= self.minimum) & (value <= self.maximum)
        log_prob[~inside_range] = float("-inf")
        return log_prob


class RateEvolution(UniformComovingVolume):
    """
    Wrapper around `UniformComovingVolume` to allow for
    arbitrary rate evolution functions. E.g., if
    `rate_function = 1 / (1 + z)`, then the distribution
    will sample values such that they occur uniform in
    source frame time.

    Args:
        rate_function: Callable that takes redshift as input
            and returns the rate evolution factor.
        *args, **kwargs: Arguments passed to `UniformComovingVolume`
            constructor.
    """

    def __init__(
        self,
        rate_function: Callable,
        *args,
        **kwargs,
    ):
        self.rate_function = rate_function
        super().__init__(*args, **kwargs)

    def _p_of_distance(self):
        """
        Compute the unnormalized probability as a function of distance
        """
        dV_dz = self.comoving_dist_grid**2 / self._hubble_function()
        # This is a tensor of ones if the distance type is redshift
        jacobian = torch.gradient(self.distance_grid, spacing=self.dz)[0]
        return dV_dz / jacobian * self.rate_function(self.z_grid)
