import torch

from ml4gw.transforms.integrator import LeakyIntegrator, TophatIntegrator


def test_tophat_integrator():
    inference_sample_rate = 10
    integration_length = 1
    integrator = TophatIntegrator(inference_sample_rate, integration_length)

    y = torch.ones(100)
    out = integrator(y)
    assert out.shape == y.shape

    start = int(integration_length * inference_sample_rate)
    torch.testing.assert_close(out[start:], torch.ones_like(out[start:]))

    y = torch.zeros(50)
    out = integrator(y)
    torch.testing.assert_close(out, torch.zeros_like(out))

    integrator = TophatIntegrator(
        inference_sample_rate=1, integration_length=3
    )

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
        dtype=y.dtype,
        device=y.device,
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
