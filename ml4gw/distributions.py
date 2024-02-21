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


class Uniform:
    """
    Sample uniformly between `low` and `high`.
    """

    def __init__(self, low: float = 0, high: float = 1) -> None:
        self.low = low
        self.high = high

    def __call__(self, N: int) -> torch.Tensor:
        return self.low + torch.rand(size=(N,)) * (self.high - self.low)


class Cosine:
    """
    Sample from a raised Cosine distribution between
    `low` and `high`. Based on the implementation from
    bilby documented here:
    https://lscsoft.docs.ligo.org/bilby/api/bilby.core.prior.analytical.Cosine.html  # noqa
    """

    def __init__(
        self, low: float = -math.pi / 2, high: float = math.pi / 2
    ) -> None:
        self.low = low
        self.norm = 1 / (math.sin(high) - math.sin(low))

    def __call__(self, N: int) -> torch.Tensor:
        """
        Implementation lifted from
        https://lscsoft.docs.ligo.org/bilby/_modules/bilby/core/prior/analytical.html#Cosine # noqa
        """
        u = torch.rand(size=(N,))
        return torch.arcsin(u / self.norm + math.sin(self.low))


class CosineDistribution(dist.Distribution):
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
        self.low = low
        self.norm = 1 / (math.sin(high) - math.sin(low))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        u = torch.rand(sample_shape)
        return torch.arcsin(u / self.norm + math.sin(self.low))

    def log_prob(self, value):
        value = torch.as_tensor(value)
        inside_range = (value >= self.low) & (value <= self.high)
        return value.cos().log() * inside_range


class SineDistribution(dist.TransformedDistribution):
    """
    Sine distribution based on
    ``torch.distributions.TransformedDistribution``.
    """

    def __init__(
        self, low: float = 0, high: float = math.pi, validate_args=None
    ):
        base_dist = CosineDistribution(
            low - math.pi / 2, high - math.pi / 2, validate_args
        )
        super().__init__(
            base_dist,
            [
                dist.AffineTransform(
                    loc=math.pi / 2,
                    scale=1,
                )
            ],
            validate_args=validate_args,
        )


class LogNormal:
    """
    Sample from a log normal distribution with the
    specified `mean` and standard deviation `std`.
    If a `low` value is specified, values sampled
    lower than this will be clipped to `low`.
    """

    def __init__(
        self, mean: float, std: float, low: Optional[float] = None
    ) -> None:
        self.sigma = math.log((std / mean) ** 2 + 1) ** 0.5
        self.mu = 2 * math.log(mean / (mean**2 + std**2) ** 0.25)
        self.low = low

    def __call__(self, N: int) -> torch.Tensor:

        u = self.mu + torch.randn(N) * self.sigma
        x = torch.exp(u)

        if self.low is not None:
            x = torch.clip(x, self.low)
        return x


class LogUniform(Uniform):
    """
    Sample from a log uniform distribution
    """

    def __init__(self, low: float, high: float) -> None:
        super().__init__(math.log(low), math.log(high))

    def __call__(self, N: int) -> torch.Tensor:
        u = super().__call__(N)
        return torch.exp(u)


class PowerLaw:
    """
    Sample from a power law distribution,
    .. math::
        p(x) \approx x^{-\alpha}.

    Index alpha must be greater than 1.
    This could be used, for example, as a universal distribution of
    signal-to-noise ratios (SNRs) from uniformly volume distributed
    sources
    .. math::

       p(\rho) = 3*\rho_0^3 / \rho^4

    where :math:`\rho_0` is a representative minimum SNR
    considered for detection. See, for example,
    `Schutz (2011) <https://arxiv.org/abs/1102.5421>`_.
    """

    def __init__(
        self, x_min: float, x_max: float = float("inf"), alpha: float = 2
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.alpha = alpha

        self.normalization = x_min ** (-self.alpha + 1)
        self.normalization -= x_max ** (-self.alpha + 1)

    def __call__(self, N: int) -> torch.Tensor:
        u = torch.rand(N)
        u *= self.normalization
        samples = self.x_min ** (-self.alpha + 1) - u
        samples = torch.pow(samples, -1.0 / (self.alpha - 1))
        return samples


class PowerLawDistribution(dist.TransformedDistribution):
    """
    Power Law distribution based on
    ``torch.distributions.TransformedDistribution``.
    Use, for example, ``index=2`` for uniform in Euclidean volume.
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
