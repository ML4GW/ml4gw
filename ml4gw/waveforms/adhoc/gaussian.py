import torch
from torch import Tensor

from ml4gw.types import BatchTensor

class Gaussian(torch.nn.Module):
    """
    Callable class for generating 'LALInference-style' Gaussian waveforms.

    This is a direct analog of the XLALInferenceBurstGaussian code
    in LALInference, using 'duration' as the standard deviation sigma
    and normalizing to produce the requested hrss.  The length of the
    times array is simply (sample_rate*duration), centered at 0.0;
    feel free to adjust this to replicate the exact ±6 sigma used
    in LAL if desired.
    """

    def __init__(self, sample_rate: float, duration: float):
        """
        Args:
            sample_rate: Sample rate (Hz).
            duration: The standard deviation (sigma) of the Gaussian (seconds).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration  # Interpreted as sigma in LAL code

        # Construct a time axis from -duration/2 to +duration/2
        num_samples = int(sample_rate * duration)
        # You could do the ±6 sigma region by:
        #   num_samples = int(12.0 * duration * sample_rate) + 1
        # but here we mimic SineGaussian style of "duration" total.
        times = torch.arange(num_samples, dtype=torch.float64) / sample_rate
        times -= 0.5 * duration

        self.register_buffer("times", times)

    def forward(
        self,
        hrss: BatchTensor,
        polarization: BatchTensor,
        eccentricity: BatchTensor,
    ):
        """
        Generate the Gaussian burst in plus and cross polarizations
        following LALInference's normalization.

        Args:
            hrss:
                The requested root-sum-square amplitude.
            polarization:
                Polarization angle (radians). LAL uses cos(pol) for hplus,
                sin(pol) for hcross.
            eccentricity:
                Ignored in the LAL code, but included here to match
                the parameter signature.

        Returns:
            cross, plus: Tensors of shape (batch, num_samples).
        """
        # Make sure everything is in a common shape
        hrss = hrss.view(-1, 1)
        polarization = polarization.view(-1, 1)
        eccentricity = eccentricity.view(-1, 1)  # not used

        # As in LALInference:
        #   hplusrss  = hrss * cos(polarization)
        #   hcrossrss = hrss * sin(polarization)
        hplusrss = hrss * torch.cos(polarization)
        hcrossrss = hrss * torch.sin(polarization)

        # LAL code uses sqrt(duration) in the denominator
        # and FRTH_2_Pi = 2 / sqrt(pi)
        FRTH_2_Pi = 2.0 / torch.sqrt(torch.pi)
        sigma = self.duration ** 0.5

        # "peak" amplitudes:
        h0_plus = hplusrss / sigma * FRTH_2_Pi
        h0_cross = hcrossrss / sigma * FRTH_2_Pi

        # Evaluate Gaussian factor, exp(-t^2 / sigma^2):
        # Note that the LAL code does exp( - (t^2) / (duration^2) )
        # which in the original is exp(-t^2 / (sigma^2)).
        t2 = self.times ** 2  # shape [num_samples]
        t2 = t2.unsqueeze(0)  # shape [1, num_samples], so we can broadcast

        fac = torch.exp(-t2 / (self.duration ** 2))

        # Multiply out:
        plus = h0_plus * fac
        cross = h0_cross * fac

        # Convert to original dtype
        dtype = hrss.dtype
        plus = plus.to(dtype)
        cross = cross.to(dtype)

        # Return cross, plus for consistency with your SineGaussian code
        return cross, plus