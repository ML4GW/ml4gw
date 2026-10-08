import pytest
import torch

from ml4gw.nn.norm import (
    GroupNorm1D,
    GroupNorm1DGetter,
    GroupNorm2D,
    GroupNorm2DGetter,
    NormLayer,
)

# each class with the input shape it normalizes, minus the channel dim
NORMS = [(GroupNorm1D, (1024,)), (GroupNorm2D, (32, 32))]


@pytest.fixture(params=NORMS, ids=["1d", "2d"])
def norm_and_shape(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def factor(request):
    return request.param


@pytest.fixture(params=[None, 1, 2, 3, 4])
def num_groups(request):
    return request.param


def test_num_groups(norm_and_shape):
    norm_cls, _ = norm_and_shape

    # one group per channel by default
    assert norm_cls(4).num_groups == 4

    # more groups than channels are capped at one group per channel
    assert norm_cls(4, 16).num_groups == 4

    with pytest.raises(ValueError):
        norm_cls(5, 3)


def test_matches_torch(norm_and_shape, num_groups, factor):
    norm_cls, shape = norm_and_shape
    num_channels = factor * (num_groups or 1)
    norm = norm_cls(num_channels, num_groups)

    # random parameters, so there's something to compare
    ref = torch.nn.GroupNorm(norm.num_groups, num_channels)
    torch.nn.init.normal_(ref.weight)
    torch.nn.init.normal_(ref.bias)
    with torch.no_grad():
        norm.weight.copy_(ref.weight.reshape(norm.weight.shape))
        norm.bias.copy_(ref.bias.reshape(norm.bias.shape))

    x = torch.randn(8, num_channels, *shape)
    torch.testing.assert_close(norm(x), ref(x), rtol=1e-4, atol=1e-4)


def test_group_norm_1d_input_shape():
    norm = GroupNorm1D(4)
    with pytest.raises(ValueError):
        norm(torch.randn(4, 1024))
    with pytest.raises(ValueError):
        norm(torch.randn(1, 4, 16, 1024))


def test_norm_getters():
    with pytest.warns(DeprecationWarning, match="GroupNorm1D"):
        getter = GroupNorm1DGetter(groups=2)
    norm = getter(8)
    assert isinstance(norm, GroupNorm1D)

    with pytest.warns(DeprecationWarning):
        getter_none = GroupNorm1DGetter()
    norm_none = getter_none(4)
    assert isinstance(norm_none, GroupNorm1D)

    with pytest.warns(DeprecationWarning, match="GroupNorm2D"):
        getter2d = GroupNorm2DGetter(groups=2)
    norm2d = getter2d(8)
    assert isinstance(norm2d, torch.nn.GroupNorm)

    with pytest.warns(DeprecationWarning):
        getter2d_none = GroupNorm2DGetter()
    norm2d_none = getter2d_none(4)
    assert isinstance(norm2d_none, torch.nn.GroupNorm)


@pytest.mark.parametrize(
    "class_path,init_args,expected",
    [
        ("ml4gw.nn.norm.GroupNorm1D", {"num_groups": 4}, GroupNorm1D),
        ("ml4gw.nn.norm.GroupNorm2D", {"num_groups": 4}, GroupNorm2D),
    ],
)
def test_norm_layer_jsonargparse(class_path, init_args, expected):
    jsonargparse = pytest.importorskip("jsonargparse")

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--norm_layer", type=NormLayer)
    cfg = parser.parse_object(
        {"norm_layer": {"class_path": class_path, "init_args": init_args}}
    )
    # instantiate_classes is deprecated from jsonargparse 4.49
    instantiate = getattr(parser, "instantiate", parser.instantiate_classes)
    norm_layer = instantiate(cfg).norm_layer

    norm = norm_layer(16)
    assert isinstance(norm, expected)
    assert (norm.num_channels, norm.num_groups) == (16, 4)
