"""
This module contains a variety of data transformation classes,
including objects to calculate spectral densities, whiten data,
and compute Q-transforms.
"""

from .decimator import Decimator
from .iirfilter import IIRFilter
from .pearson import ShiftedPearsonCorrelation
from .qtransform import QScan, SingleQTransform
from .scaler import ChannelWiseScaler
from .snr_rescaler import SnrRescaler
from .spectral import SpectralDensity
from .spectrogram import MultiResolutionSpectrogram
from .spline_interpolation import SplineInterpolate1D, SplineInterpolate2D
from .waveforms import WaveformProjector, WaveformSampler
from .whitening import FixedWhiten, Whiten
