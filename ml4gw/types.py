from torchtyping import TensorType

WaveformTensor = TensorType["batch", "num_ifos", "time"]
PSDTensor = TensorType["num_ifos", "frequency"]
ScalarTensor = TensorType["batch"]
VectorGeometry = TensorType["batch", "space"]
TensorGeometry = TensorType["batch", "space", "space"]
NetworkVertices = TensorType["num_ifos", 3]
NetworkDetectorTensors = TensorType["num_ifos", 3, 3]
TimeSeriesTensor = TensorType["num_channels", "time"]
