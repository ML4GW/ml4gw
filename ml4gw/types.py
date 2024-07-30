from jaxtyping import Array, Float

WaveformTensor = Float[Array, "batch num_ifos time"]
PSDTensor = Float[Array, "num_ifos frequency"]
ScalarTensor = Float[Array, "batch"]
VectorGeometry = Float[Array, "batch space"]
TensorGeometry = Float[Array, "batch space space"]
NetworkVertices = Float[Array, "num_ifos 3"]
NetworkDetectorTensors = Float[Array, "num_ifos 3 3"]
TimeSeriesTensor = Float[Array, "num_channels time"]
