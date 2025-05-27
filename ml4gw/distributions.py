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
            raise ValueError("Index of 0 is the same as Uniform")
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
