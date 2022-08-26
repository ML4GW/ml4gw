import bilby
import numpy as np
import pytest
import torch
from lal import GreenwichMeanSiderealTime

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


@pytest.fixture(params=[["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def ifos(request):
    return request.param


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture(params=[512, 1024])
def sample_rate(request):
    return request.param


@pytest.fixture
def waveform_duration():
    return 8


@pytest.fixture
def data(batch_size, sample_rate, waveform_duration):
    gps_times = [float(1234567890 + i) for i in range(batch_size)]
    gmst = [GreenwichMeanSiderealTime(i) % (2 * np.pi) for i in gps_times]
    gmst = np.array(gmst)

    ra = np.random.uniform(0, 2 * np.pi, size=(batch_size,))
    dec = np.random.uniform(-np.pi / 2, np.pi / 2, size=(batch_size,))
    psi = np.random.uniform(0, 2 * np.pi, size=(batch_size,))
    phi = ra - gmst

    t = np.arange(0.0, waveform_duration, 1 / sample_rate)
    assert len(t) == (waveform_duration * sample_rate)

    plus = np.stack([np.sin(20 * 2 * np.pi * t)] * batch_size)
    cross = np.stack([0.5 * np.sin(20 * 2 * np.pi * t)] * batch_size)
    return ra, dec, psi, phi, gps_times, plus, cross


@pytest.fixture
def bilby_get_ifo_response(ifos, batch_size):
    ifos = [bilby.gw.detector.get_empty_interferometer(i) for i in ifos]

    def func(ra, dec, psi, geocent_time, modes):
        responses = np.zeros((batch_size, len(modes), len(ifos)))
        for i, (r, d, p, t) in enumerate(zip(ra, dec, psi, geocent_time)):
            for j, ifo in enumerate(ifos):
                for k, mode in enumerate(modes):
                    responses[i, k, j] = ifo.antenna_response(r, d, t, p, mode)
        return responses

    return func


def test_compute_antenna_responses(
    ifos,
    batch_size,
    sample_rate,
    waveform_duration,
    bilby_get_ifo_response,
    data,
):
    ra, dec, psi, phi, gps_times, plus, cross = data
    expected = bilby_get_ifo_response(
        ra, dec, psi, gps_times, ["plus", "cross"]
    )

    phi = torch.tensor(phi)
    dec = torch.tensor(dec)
    psi = torch.tensor(psi)
    tensors, vertices = injection.get_ifo_geometry(*ifos)
    tensors = tensors.type(torch.float64)

    result = injection.compute_antenna_responses(
        np.pi / 2 - dec, psi, phi, tensors, ["plus", "cross"]
    )
    assert result.shape == (batch_size, 2, len(ifos))
    assert np.isclose(result, expected, rtol=1e-6).all()


@pytest.fixture
def bilby_get_projections(
    ifos, batch_size, sample_rate, waveform_duration, bilby_get_ifo_response
):
    ifos = [bilby.gw.detector.get_empty_interferometer(i) for i in ifos]
    waveform_size = int(waveform_duration * sample_rate)

    def func(ra, dec, psi, geocent_time, **polarizations):
        output = np.zeros((batch_size, len(ifos), waveform_size))

        modes = list(polarizations.keys())
        responses = bilby_get_ifo_response(ra, dec, psi, geocent_time, modes)
        responses = responses.transpose(2, 1, 0)

        for i, mode in enumerate(modes):
            for j, response in enumerate(responses):
                waveform = (polarizations[mode].T * response[i]).T
                output[:, j] += waveform
        return output

    return func


def test_project_waveforms(
    ifos,
    batch_size,
    sample_rate,
    waveform_duration,
    bilby_get_projections,
    data,
):
    ra, dec, psi, phi, gps_times, plus, cross = data
    expected = bilby_get_projections(
        ra, dec, psi, gps_times, plus=plus, cross=cross
    )

    phi = torch.tensor(phi)
    dec = torch.tensor(dec)
    psi = torch.tensor(psi)
    plus = torch.tensor(plus)
    cross = torch.tensor(cross)
    tensors, vertices = injection.get_ifo_geometry(*ifos)
    tensors = tensors.type(torch.float64)

    responses = injection.compute_antenna_responses(
        np.pi / 2 - dec, psi, phi, tensors, ["plus", "cross"]
    )
    result = injection.project_waveforms(responses, plus=plus, cross=cross)
    assert result.shape == (
        batch_size,
        len(ifos),
        sample_rate * waveform_duration,
    )
    assert np.isclose(result, expected, rtol=1e-6).all()


@pytest.fixture
def bilby_shift_projections(
    ifos,
    batch_size,
    sample_rate,
    waveform_duration,
):
    ifos = [bilby.gw.detector.get_empty_interferometer(i) for i in ifos]
    waveform_size = int(waveform_duration * sample_rate)

    def do_shift(ifo, ra, dec, geocent_time, response):
        shift = ifo.time_delay_from_geocenter(ra, dec, geocent_time)
        shift *= sample_rate
        shift += waveform_size // 2
        response = np.roll(response, int(shift))
        return response

    def func(ra, dec, geocent_time, responses):
        output = np.zeros_like(responses)
        for i, (r, d, t, sample) in enumerate(
            zip(ra, dec, geocent_time, responses)
        ):
            for j, (response, ifo) in enumerate(zip(sample, ifos)):
                shifted = do_shift(ifo, r, d, t, response)
                output[i, j] = shifted
        return output

    return func


def test_shift_projections(
    ifos,
    batch_size,
    sample_rate,
    waveform_duration,
    bilby_shift_projections,
    data,
):
    ra, dec, psi, phi, gps_times, plus, cross = data
    projections = np.random.randn(
        batch_size, len(ifos), int(waveform_duration * sample_rate)
    )
    expected = bilby_shift_projections(ra, dec, gps_times, projections)

    projections = torch.tensor(projections)
    phi = torch.tensor(phi)
    dec = torch.tensor(dec)
    psi = torch.tensor(psi)
    tensors, vertices = injection.get_ifo_geometry(*ifos)
    vertices = vertices.type(torch.float64)

    result = injection.shift_projections(
        projections, sample_rate, np.pi / 2 - dec, psi, phi, vertices
    )
    assert result.shape == projections.shape
    assert np.isclose(result, expected, rtol=1e-6).all()


@pytest.fixture
def bilby_project_raw_gw(bilby_get_projections, bilby_shift_projections):
    def func(ra, dec, psi, gps_times, **polarizations):
        projections = bilby_get_projections(
            ra, dec, psi, gps_times, **polarizations
        )
        return bilby_shift_projections(ra, dec, gps_times, projections)

    return func


def test_project_raw_gw(
    ifos,
    batch_size,
    sample_rate,
    waveform_duration,
    bilby_project_raw_gw,
    data,
):
    ra, dec, psi, phi, gps_times, plus, cross = data
    expected = bilby_project_raw_gw(
        ra, dec, psi, gps_times, plus=plus, cross=cross
    )

    phi = torch.tensor(phi)
    dec = torch.tensor(dec)
    psi = torch.tensor(psi)
    plus = torch.tensor(plus)
    cross = torch.tensor(cross)
    tensors, vertices = injection.get_ifo_geometry(*ifos)
    tensors = tensors.type(torch.float64)
    vertices = vertices.type(torch.float64)

    result = injection.project_raw_gw(
        sample_rate, dec, psi, phi, tensors, vertices, plus=plus, cross=cross
    )
    assert result.shape == (
        batch_size,
        len(ifos),
        waveform_duration * sample_rate,
    )
    assert np.isclose(result, expected, rtol=1e-6).all()
