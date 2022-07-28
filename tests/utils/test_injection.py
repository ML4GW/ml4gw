import numpy as np
import torch

from ml4gw.utils import injection


def test_outer():
    x = torch.randn(3, 10)
    y = torch.randn(3, 10)
    output = injection.outer(x, y)

    x, y = x.cpu().numpy(), y.cpu().numpy()
    for i, matrix in enumerate(output.cpu().numpy()):
        for j, row in enumerate(matrix):
            for k, value in enumerate(row):
                assert value == x[j, i] * y[k, i], (i, j, k)


def test_project_raw_gw():
    batch_size = 8
    sample_rate = 1024

    dec = torch.randn(size=(batch_size,))
    psi = torch.randn(size=(batch_size,))
    phi = torch.randn(size=(batch_size,))
    tensors, vertices = injection.get_ifo_geometry("H1", "L1")

    t = torch.arange(0.0, 8.0, 1 / sample_rate)
    assert len(t) == (8 * sample_rate)

    plus = torch.stack([torch.sin(20 * 2 * np.pi * t)] * batch_size)
    cross = torch.stack([0.5 * torch.sin(20 * 2 * np.pi * t)] * batch_size)

    result = injection.project_raw_gw(
        sample_rate, dec, psi, phi, tensors, vertices, plus=plus, cross=cross
    )
    assert result.shape == (batch_size, 2, len(t))
    # TODO: compare to bilby output. How to do this without
    # having to rely on lalsuite as a dependency?
