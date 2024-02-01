import h5py  # noqa
import pytest
import torch

from ml4gw.nn.resnet.resnet_2d import (
    BasicBlock,
    Bottleneck,
    BottleneckResNet2D,
    ResNet2D,
    conv1,
)


@pytest.fixture(params=[3, 7, 8])
def kernel_size(request):
    return request.param


@pytest.fixture(params=[10, 50])
def spectrogram_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request):
    return request.param


@pytest.fixture(params=[2, 4])
def inplanes(request):
    return request.param


@pytest.fixture(params=[1, 2])
def classes(request):
    return request.param


@pytest.fixture(params=[BasicBlock, Bottleneck])
def block(request):
    return request.param


def test_blocks(block, kernel_size, stride, spectrogram_size, inplanes):
    # TODO: test dilation for bottleneck
    planes = 4

    if stride > 1 or inplanes != planes * block.expansion:
        downsample = conv1(inplanes, planes * block.expansion, stride)
    else:
        downsample = None

    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            block = block(
                inplanes, planes, kernel_size, stride, downsample=downsample
            )
        return

    block = block(inplanes, planes, kernel_size, stride, downsample=downsample)
    x = torch.randn(8, inplanes, spectrogram_size, spectrogram_size)
    y = block(x)

    assert len(y.shape) == 4
    assert y.shape[1] == planes * block.expansion
    assert y.shape[2] == spectrogram_size // stride
    assert y.shape[3] == spectrogram_size // stride


@pytest.fixture(params=[1, 2, 3])
def in_channels(request):
    return request.param


@pytest.fixture(params=[[2, 2, 2, 2], [2, 4, 4], [3, 4, 6, 3]])
def layers(request):
    return request.param


@pytest.fixture(params=[None, "stride", "dilation"])
def stride_type(request):
    return request.param


@pytest.fixture(params=[BottleneckResNet2D, ResNet2D])
def architecture(request):
    return request.param


def test_resnet(
    architecture,
    kernel_size,
    layers,
    classes,
    in_channels,
    spectrogram_size,
    stride_type,
):
    if kernel_size % 2 == 0:
        with pytest.raises(ValueError):
            nn = ResNet2D(in_channels, layers, classes, kernel_size)
        return

    if stride_type is not None:
        stride_type = [stride_type] * (len(layers) - 1)

    if (
        stride_type is not None
        and stride_type[0] == "dilation"
        and architecture == ResNet2D
    ):
        with pytest.raises(NotImplementedError):
            nn = architecture(
                in_channels,
                layers,
                classes,
                kernel_size,
                stride_type=stride_type,
            )
        return

    nn = architecture(
        in_channels, layers, classes, kernel_size, stride_type=stride_type
    )
    x = torch.randn(8, in_channels, spectrogram_size, spectrogram_size)
    y = nn(x)
    assert y.shape == (8, classes)

    with pytest.raises(ValueError):
        stride_type = ["stride"] * len(layers)
        nn = architecture(
            in_channels, layers, kernel_size, stride_type=stride_type
        )
    with pytest.raises(ValueError):
        stride_type = ["strife"] * (len(layers) - 1)
        nn = architecture(
            in_channels, layers, kernel_size, stride_type=stride_type
        )
