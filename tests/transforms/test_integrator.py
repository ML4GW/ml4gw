import pytest
import torch

from ml4gw.transforms.integrator import LeakyIntegrator, TophatIntegrator


def test_tophat_integrator():
    sample_rate = 10
    integration_length = 1
    integrator = TophatIntegrator(sample_rate, integration_length)

    y = torch.ones(100)
    out = integrator(y)
    assert out.shape == y.shape

    start = int(sample_rate * integration_length)
    torch.testing.assert_close(out[start:], torch.ones_like(out[start:]))

    y = torch.zeros(50)
    out = integrator(y)
    torch.testing.assert_close(out, torch.zeros_like(out))

    integrator = TophatIntegrator(sample_rate=1, integration_length=3)

    y = torch.randn(5)
    out = integrator(y)

    expected = torch.tensor(
        [
            y[0] / 4,
            (y[0] + y[1]) / 4,
            (y[0] + y[1] + y[2]) / 4,
            (y[0] + y[1] + y[2] + y[3]) / 4,
            (y[1] + y[2] + y[3] + y[4]) / 4,
        ],
    )

    torch.testing.assert_close(out, expected)


def test_leaky_integrator():
    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=1.0,
        integrate_value="count",
        lower_bound=0.0,
    )

    y = torch.tensor([0.1, 0.6, 0.7, 0.2, 0.8])
    out = integrator(y)

    torch.testing.assert_close(
        out,
        torch.tensor(
            [0.0, 1.0, 2.0, 1.0, 2.0], dtype=out.dtype, device=out.device
        ),
    )

    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=0.5,
        integrate_value="score",
        lower_bound=0.0,
    )

    y = torch.tensor([0.6, 0.7, 0.2])
    out = integrator(y)

    torch.testing.assert_close(
        out,
        torch.tensor([0.6, 1.3, 0.8], dtype=out.dtype, device=out.device),
    )

    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=1.0,
        integrate_value="count",
        lower_bound=0.0,
    )

    y = torch.tensor([0.6, 0.7, 0.4, 0.6, 0.7, 0.7, 0.4])
    out = integrator(y)

    torch.testing.assert_close(
        out,
        torch.tensor(
            [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0],
            dtype=out.dtype,
            device=out.device,
        ),
    )


@pytest.mark.parametrize(
    "shape",
    [(30,), (20, 30), (20, 30, 100)],
)
@pytest.mark.parametrize(
    "integrator",
    [
        LeakyIntegrator(
            threshold=0.5,
            decay=0.5,
            integrate_value="count",
            lower_bound=0.0,
        ),
        TophatIntegrator(
            sample_rate=5,
            integration_length=1,
        ),
    ],
)
def test_integrator_dimensions(integrator, shape):
    y = torch.randn(shape)
    out = integrator(y)
    assert out.shape == y.shape

    y_flat = y.reshape(-1, shape[-1])
    out_flat = out.reshape(-1, shape[-1])

    for i in range(y_flat.shape[0]):
        output = integrator(y_flat[i])
        torch.testing.assert_close(out_flat[i], output)
