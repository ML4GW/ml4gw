"""
This module defines common types used for type
annotation throughout the package
"""

from jaxtyping import Complex, Float, Inexact
from torch import Tensor

WaveformTensor = Float[Tensor, "batch num_ifos time"]
BatchTensor = Float[Tensor, "batch"]
VectorGeometry = Float[Tensor, "*batch space"]
TensorGeometry = Float[Tensor, "*batch space space"]
NetworkVertices = Float[Tensor, "num_ifos 3"]
NetworkDetectorTensors = Float[Tensor, "num_ifos 3 3"]


TimeSeries1d = Float[Tensor, "time"]
TimeSeries2d = Float[TimeSeries1d, "channel"]
TimeSeries3d = Float[TimeSeries2d, "batch"]
TimeSeries1to3d = TimeSeries1d | TimeSeries2d | TimeSeries3d

FrequencySeries1d = Float[Tensor, "frequency"]
FrequencySeries2d = Float[FrequencySeries1d, "channel"]
FrequencySeries3d = Float[FrequencySeries2d, "batch"]
FrequencySeries1to3d = (
    FrequencySeries1d | FrequencySeries2d | FrequencySeries3d
)

# An ASD, which broadcasts against (batch, channel, time) data
ASDTensor = (
    FrequencySeries1d
    | Float[Tensor, "#channel frequency"]
    | Float[Tensor, "#batch #channel frequency"]
)

# A power spectral density is real, but a cross spectral density is complex
SpectrumTensor = (
    Inexact[Tensor, " frequency"]
    | Inexact[Tensor, "channel frequency"]
    | Inexact[Tensor, "batch channel frequency"]
)

# A batch of complex frequency-domain waveforms
FrequencyDomainWaveform = Complex[Tensor, "batch frequency"]

# Whitened data. Whitening crops the edges, so its time dimension is shorter
WhitenedTimeSeries3d = Float[Tensor, "batch channel time2"]
