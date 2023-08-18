import pytest
import torch

from ml4gw.transforms.scaler import ChannelWiseScaler


def test_scaler_1d():
    scaler = ChannelWiseScaler()
    assert len(list(scaler.buffers())) == 2
    assert scaler.mean.shape == (1,)
    assert scaler.std.shape == (1,)

    # test fitting
    background = torch.arange(1, 11).type(torch.float32)
    scaler.fit(background)

    mean, std = 5.5, (99 / 12) ** 0.5
    assert scaler.mean.item() == pytest.approx(mean)
    assert scaler.std.item() == pytest.approx(std)

    # test fit shape checks
    with pytest.raises(AssertionError):
        scaler.fit(background[None])
    with pytest.raises(ValueError):
        scaler.fit(background[None, None])

    # test forward
    x = torch.arange(20).reshape(2, 10).type(torch.float32) + 1
    y = scaler(x)
    for i in range(2):
        expected = i * 10 + torch.arange(1, 11)
        expected = (expected - mean) / std
        assert (y[i] == expected).all().item()

    # now reverse it
    y = scaler(y, reverse=True)
    assert torch.isclose(x, y, rtol=1e-6).all().item()


def test_scaler_2d():
    num_channels = 4
    scaler = ChannelWiseScaler(num_channels)
    assert len(list(scaler.buffers())) == 2
    assert scaler.mean.shape == (num_channels, 1)
    assert scaler.std.shape == (num_channels, 1)

    # test fitting
    background = (
        torch.arange(num_channels * 10)
        .reshape(num_channels, 10)
        .type(torch.float32)
        + 1
    )
    scaler.fit(background)

    std = (99 / 12) ** 0.5
    for i in range(num_channels):
        mean = ((2 * i + 1) * 10 + 1) / 2
        assert scaler.mean[i, 0] == pytest.approx(mean)
        assert scaler.std[i, 0] == pytest.approx(std)

    # test fit shape checks
    with pytest.raises(AssertionError):
        scaler.fit(background[0])
    with pytest.raises(ValueError):
        scaler.fit(background[None, None])

    # test forward with 2D
    x = torch.arange(num_channels * 10).type(torch.float32)
    x = x.reshape(num_channels, 10).type(torch.float32) + 1
    y = scaler(x)
    expected = (torch.arange(1, 11) - 5.5) / std
    assert torch.isclose(y, expected, rtol=1e-6).all().item()

    # now reverse it
    y = scaler(y, reverse=True)
    assert torch.isclose(x, y, rtol=1e-6).all().item()

    # test forward with 3D
    x = torch.stack([x, x + 1])
    y = scaler(x)
    expected = torch.stack([expected, expected + 1 / std])
    assert torch.isclose(y.transpose(1, 0), expected, rtol=1e-6).all().item()

    # now reverse
    y = scaler(y, reverse=True)
    assert torch.isclose(x, y, rtol=1e-6).all().item()


def test_scaler_save_and_load(tmp_path):
    scaler = ChannelWiseScaler()
    background = torch.arange(1, 11).type(torch.float32)

    scaler.fit(background)
    assert scaler.built
    mean, std = 5.5, (99 / 12) ** 0.5

    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(scaler.state_dict(), tmp_path / "scaler.pt")

    scaler = ChannelWiseScaler()
    assert not scaler.built
    assert (scaler.mean == 0).all().item()
    assert (scaler.std == 1).all().item()

    scaler.load_state_dict(torch.load(tmp_path / "scaler.pt"))
    assert scaler.built
    assert scaler.mean == mean
    assert scaler.std == std

    scaler = ChannelWiseScaler(2)
    with pytest.raises(RuntimeError):
        scaler.load_state_dict(torch.load(tmp_path / "scaler.pt"))
