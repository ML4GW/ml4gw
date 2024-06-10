import torch
from torch import Tensor
import numpy as np
from ml4gw.types import ScalarTensor

c = 3.0e8 # speed of ligt
G = 6.67430e-11 # gravitational constant

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
        frequency: ScalarTensor, #Hz
        quality: ScalarTensor, 
        epsilon: ScalarTensor,  
        phase: ScalarTensor, #rad
        inclination: ScalarTensor, #rad
        distance: ScalarTensor, #Mpc
    ):
        """
        Generate ringdown waveform based on the damped sinusoid equation.

        Args:
            frequency: 
                Central frequency of the ringdown waveform
            quality:
                Quality factor of the ringdown waveform
            epsilon: 
                Fraction of black hole's mass radiated as gravitational waves
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
        epsilon = epsilon.view(-1, 1)
        phase = phase.view(-1, 1)
        inclination = inclination.view(-1, 1)
        distance = distance.view(-1, 1)

        distance = distance*(3.086e22) #conveting to meters from MPC

        # ensure all inputs are on the same device
        pi = torch.tensor([torch.pi], device=frequency.device)

        # Calculate spin 
        spin = 1 - (2 / quality)**(20 / 9)
        # Calculate mass 
        mass = (1 / (2 * pi)) * (c**3 / (G * frequency)) * (1 - 0.63 * (2 / quality)**(2 / 3))

        # Calculate amplitude 
        F_Q = 1 + (1.34 / quality**2)
        g_a = (1 + 0.1 * spin**2)
        amplitude = np.sqrt(5 / 2) * epsilon * (6.67430e-11 * mass / (3e8**2)) * quality**(-0.5) * F_Q**(-0.5) * g_a**(-0.5)

        # calculate cosines with inclination
        cos_i = torch.cos(inclination)
        cos_i2 = cos_i**2
        sin_i = torch.sin(inclination)

        # Precompute exponent and phase terms
        exp_term = torch.exp(-pi * frequency * self.times / quality)
        phase_term = 2 * pi * frequency * self.times + phase

        a_plus = (amplitude/distance) * (1 + cos_i2) * exp_term
        a_cross = (amplitude/distance) * (2 * sin_i) * exp_term

        h_plus = a_plus * torch.cos(phase_term)
        h_cross = a_cross * torch.sin(phase_term)

        # ensure the dtype is double
        h_plus = h_plus.double()
        h_cross = h_cross.double()

        return h_cross, h_plus
