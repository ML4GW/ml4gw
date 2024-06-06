import torch
from torch import Tensor
from ml4gw.types import ScalarTensor

class Ringdown(torch.nn.Module):
    """
    Callable class for generating ringdown waveforms.

    Args:
        sample_rate: Sample rate of waveform
        duration: Duration of waveform
    """

    def __init__(self, sample_rate: float, duration: float):
        super().__init__()
        # determine times based on requested duration and sample rate
        # and shift so that the waveform is centered at t=0

        num = int(duration * sample_rate)
        times = torch.arange(num, dtype=torch.float64) / sample_rate
        times -= duration / 2.0

        self.register_buffer("times", times)

    def __call__(
        self,
        frequency: ScalarTensor,
        quality: ScalarTensor,
        amplitude: ScalarTensor,
        phase: ScalarTensor,
        inclination: ScalarTensor,
        #distance: ScalarTensor,  
    ):
        """
        Generate ringdown waveform based on the damped sinusoid equation.

        Args:
            frequency:
                Central frequency of the ringdown waveform
            quality:
                Quality factor of the ringdown waveform
            amplitude:
                Initial amplitude of the ringdown waveform
            phase:
                Initial phase of the ringdown waveform
            inclination:
                Inclination angle of the source.
            distance:
                Distance to the source.
        Returns:
            Tensors of cross and plus polarizations
        """

        # add dimension for calculating waveforms in batch
        frequency = frequency.view(-1, 1)
        quality = quality.view(-1, 1)
        amplitude = amplitude.view(-1, 1)
        phase = phase.view(-1, 1)
        inclination = inclination.view(-1, 1)
        #distance = distance.view(-1, 1)

        # ensure all inputs are on the same device
        pi = torch.tensor([torch.pi], device=frequency.device)

        # calculate cosines with inclination
        cos_i = torch.cos(inclination)
        cos_i2 = cos_i**2
        sin_i = torch.sin(inclination)


        # computing (A/r) times wave function h(t)
        # Original implementation based on equations 3.12 and 3.13
        # h0 = (amplitude / distance) * torch.exp(-pi * frequency * self.times / quality) * torch.cos(2 * pi * frequency * self.times - phase)
        # h_plus = (1 + cos_i2) * h0
        # h_cross = 2 * cos_i * h0

        # New implementation based on equations 7.5 and 7.6
        a_plus = amplitude * (1 + cos_i2) * torch.exp(-pi * frequency * self.times / quality)
        a_cross = amplitude * (2 * sin_i) * torch.exp(-pi * frequency * self.times / quality)

        h_plus = a_plus * torch.cos(2 * pi * frequency * self.times + phase)
        h_cross = a_cross * torch.sin(2 * pi * frequency * self.times + phase)

        # ensure the dtype is double
        h_plus = h_plus.double()
        h_cross = h_cross.double()

        return h_cross, h_plus


