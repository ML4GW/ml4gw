from jaxtyping import Float
from torch import Tensor

WaveformTensor = Float[Tensor, "batch num_ifos time"]
PSDTensor = Float[Tensor, "num_ifos frequency"]
BatchTensor = Float[Tensor, "batch"]
VectorGeometry = Float[Tensor, "batch space"]
TensorGeometry = Float[Tensor, "batch space space"]
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
