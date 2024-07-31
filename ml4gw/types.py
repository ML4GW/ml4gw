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
