import numpy as np
import pytest

from ml4gw.transforms.integrator import LeakyIntegrator, TophatIntegrator


def test_tophat_integrator():
    inference_sample_rate = 10
    integration_length = 1
    integrator = TophatIntegrator(inference_sample_rate, integration_length)

    y = np.ones(100)
    out = integrator(y)
    assert out.shape == y.shape
    np.testing.assert_allclose(
        out[integration_length * inference_sample_rate :], 1.0
    )

    y = np.zeros(50)
    out = integrator(y)
    assert np.all(out == 0.0)

    integrator = TophatIntegrator(
        inference_sample_rate=1, integration_length=3
    )
    y = np.random.rand(5)
    out = integrator(y)

    expected = np.array(
        [
            y[0] / 3,
            (y[0] + y[1]) / 3,
            (y[0] + y[1] + y[2]) / 3,
            (y[1] + y[2] + y[3]) / 3,
            (y[2] + y[3] + y[4]) / 3,
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_leaky_integrator():
    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=1.0,
        integrate_value="count",
        lower_bound=0.0,
    )

    y = np.array([0.1, 0.6, 0.7, 0.2, 0.8])
    out = integrator(y)
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 1.0, 2.0])

    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=0.5,
        integrate_value="score",
        lower_bound=0.0,
    )

    y = np.array([0.6, 0.7, 0.2])
    out = integrator(y)
    np.testing.assert_allclose(out, [0.6, 1.3, 0.8])

    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=0.0,
        integrate_value="count",
        lower_bound=0.0,
        detection_threshold=2.0,
    )

    y = np.array([0.6, 0.7, 0.4, 0.6, 0.7, 0.7, 0.4])
    out = integrator(y)
    np.testing.assert_allclose(out, [1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0])

    integrator = LeakyIntegrator(
        threshold=0.5,
        decay=1.0,
        integrate_value="invalid",
        lower_bound=0.0,
    )

    with pytest.raises(ValueError):
        integrator(np.array([1.0]))
