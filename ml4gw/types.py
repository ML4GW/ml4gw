from typing import Union

from jaxtyping import Float
from torch import Tensor

WaveformTensor = Float[Tensor, "batch num_ifos time"]
PSDTensor = Float[Tensor, "num_ifos frequency"]
ScalarTensor = Float[Tensor, "batch"]
VectorGeometry = Float[Tensor, "batch space"]
TensorGeometry = Float[Tensor, "batch space space"]
NetworkVertices = Float[Tensor, "num_ifos 3"]
NetworkDetectorTensors = Float[Tensor, "num_ifos 3 3"]
TimeSeriesTensor = Float[Tensor, "num_channels time"]


TimeSeries1d = Float[Tensor, "time"]
TimeSeries2d = Float[TimeSeries1d, "channel"]
TimeSeries3d = Float[TimeSeries2d, "batch"]
TimeSeries1to3d = Union[TimeSeries1d, TimeSeries2d, TimeSeries3d]
