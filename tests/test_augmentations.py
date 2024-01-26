from unittest.mock import patch

import pytest
import torch

from ml4gw.augmentations import SignalInverter, SignalReverser


@pytest.fixture(params=[0.0, 0.25, 0.5, 1])
def flip_prob(request):
    return request.param


@pytest.fixture
def rvs():
    return torch.Tensor([[0.0, 0.49], [0.51, 0.1], [0.9, 0.2], [0.1, 0.3]])


@pytest.fixture
def true_idx(flip_prob):
    if flip_prob in (0, 1):
        idx = [k for i in range(4) for k in [[i, j] for j in range(2)]]
        if flip_prob:
            neg_idx = idx
            pos_idx = []
        else:
            pos_idx = idx
            neg_idx = []
    else:
        if flip_prob == 0.5:
            neg_idx = [
                [0, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [3, 0],
                [3, 1],
            ]
            pos_idx = [
                [1, 0],
                [2, 0],
            ]
        else:
            neg_idx = [[0, 0], [1, 1], [2, 1], [3, 0]]
            pos_idx = [
                [0, 1],
                [1, 0],
                [2, 0],
                [3, 1],
            ]

    return neg_idx, pos_idx


def validate_augmenters(X, idx, true, false, prob):
    neg_idx, pos_idx = idx
    if neg_idx:
        neg0, neg1 = zip(*neg_idx)
        assert (X[neg0, neg1] == true).all()
    elif prob != 0:
        raise ValueError("Missing negative indices")

    if pos_idx:
        pos0, pos1 = zip(*pos_idx)
        assert (X[pos0, pos1] == false).all()
    elif prob != 1:
        raise ValueError("Missing positive indices")


def test_signal_inverter(flip_prob, rvs, true_idx):
    tform = SignalInverter(prob=0.0)
    X = torch.randn((4, 2, 8))
    Y = torch.clone(X)
    X = tform(X)
    assert (X == Y).all()

    tform = SignalInverter(prob=1.0)
    X = torch.randn((4, 2, 8))
    Y = torch.clone(X)
    X = tform(X)

    assert (X == -Y).all()

    tform = SignalInverter(flip_prob)
    X = torch.ones((4, 2, 8))
    with patch("torch.rand", return_value=rvs):
        X = tform(X)
    X = X.cpu().numpy()
    validate_augmenters(X, true_idx, -1, 1, flip_prob)


def test_signal_reverser(flip_prob, rvs, true_idx):
    tform = SignalReverser(prob=0.0)
    X = torch.randn((4, 2, 8))
    Y = torch.clone(X)
    X = tform(X)
    assert (X == Y).all()

    tform = SignalReverser(prob=1.0)
    X = torch.randn((4, 2, 8))
    Y = torch.clone(X)
    X = tform(X)
    assert (X == Y.flip(-1)).all()

    tform = SignalReverser(flip_prob)
    x = torch.arange(8)
    X = torch.stack([x] * 2)
    X = torch.stack([X] * 4)
    with patch("torch.rand", return_value=rvs):
        X = tform(X)
    X = X.cpu().numpy()
    x = x.cpu().numpy()
    validate_augmenters(X, true_idx, x[::-1], x, flip_prob)
