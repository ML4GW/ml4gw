import lal
import lalsimulation
import numpy as np
import pytest
import torch
from astropy import units as u
from scipy.signal import filtfilt, iirfilter
from torch.distributions import Uniform

from ml4gw.constants import MSUN
from ml4gw.transforms.iirfilter import IIRFilter
from ml4gw.waveforms.conversion import (
    bilby_spins_to_lalsim,
    chirp_mass_and_mass_ratio_to_components,
)

low_cutoff = 100
high_cutoff = 20
filters = ["cheby1", "cheby2", "ellip", "bessel", "butter"]
rprs = [(0.5, None), (None, 20), (0.5, 20), (None, None), (None, None)]


@pytest.fixture(params=[256, 512, 1024, 2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2, 4, 6, 8])
def order(request):
    return request.param


def test_filters_synthetic_signal(sample_rate, order):
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone_freq = 50
    noise_amplitude = 0.5

    signal = np.sin(2 * np.pi * tone_freq * t)
    noise = noise_amplitude * np.random.normal(size=t.shape)
    combined_signal = signal + noise

    slice_length = int(0.15 * sample_rate)

    for ftype, (rp, rs) in zip(filters, rprs):
        b, a = iirfilter(
            order,
            low_cutoff,
            btype="low",
            analog=False,
            output="ba",
            fs=sample_rate,
            rp=rp,
            rs=rs,
            ftype=ftype,
        )
        scipy_filtered_data_low = filtfilt(b, a, combined_signal)[
            slice_length:-slice_length
        ]

        b, a = iirfilter(
            order,
            high_cutoff,
            btype="high",
            analog=False,
            output="ba",
            fs=sample_rate,
            rp=rp,
            rs=rs,
            ftype=ftype,
        )
        scipy_filtered_data_high = filtfilt(b, a, combined_signal)[
            slice_length:-slice_length
        ]

        # test one of these with a tensor input instead of scalar Wn, rs, rps
        torch_filtered_data_low = IIRFilter(
            order,
            torch.tensor(low_cutoff),
            btype="low",
            analog=False,
            fs=sample_rate,
            rs=torch.tensor(rs) if rs is not None else None,
            rp=torch.tensor(rp) if rp is not None else None,
            ftype=ftype,
        )(torch.tensor(combined_signal).repeat(10, 1))[
            :, slice_length:-slice_length
        ].numpy()

        torch_filtered_data_high = IIRFilter(
            order,
            high_cutoff,
            btype="high",
            analog=False,
            fs=sample_rate,
            rs=rs,
            rp=rp,
            ftype=ftype,
        )(torch.tensor(combined_signal).repeat(10, 1))[
            :, slice_length:-slice_length
        ].numpy()

        # test batch processing
        for i in range(9):
            assert np.allclose(
                torch_filtered_data_low[0],
                torch_filtered_data_low[i + 1],
                atol=float(np.finfo(float).eps),
            )
            assert np.allclose(
                torch_filtered_data_high[0],
                torch_filtered_data_high[i + 1],
                atol=float(np.finfo(float).eps),
            )

        assert np.allclose(
            scipy_filtered_data_low,
            torch_filtered_data_low[0],
            atol=2e-1,
        )
        assert np.allclose(
            scipy_filtered_data_high,
            torch_filtered_data_high[0],
            atol=2e-1,
        )


N_SAMPLES = 1


@pytest.fixture()
def chirp_mass(seed_everything, request):
    dist = Uniform(5, 100)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def mass_ratio(seed_everything):
    dist = Uniform(0.125, 0.99)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def a_1(seed_everything, request):
    dist = Uniform(0, 0.90)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def a_2(seed_everything, request):
    dist = Uniform(0, 0.90)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def tilt_1(seed_everything, request):
    dist = Uniform(0, torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def tilt_2(seed_everything, request):
    dist = Uniform(0, torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def phi_12(seed_everything, request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def phi_jl(seed_everything, request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def distance(seed_everything, request):
    dist = Uniform(100, 3000)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def theta_jn(seed_everything, request):
    dist = Uniform(0, torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture()
def phase(seed_everything, request):
    dist = Uniform(0, 2 * torch.pi)
    return dist.sample(torch.Size((N_SAMPLES,)))


@pytest.fixture(params=[20, 40])
def f_ref(request):
    return request.param


def test_filters_phenom_signal(
    sample_rate,
    order,
    chirp_mass,
    mass_ratio,
    distance,
    phase,
    f_ref,
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
):
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )
    (
        inclination,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
    ) = bilby_spins_to_lalsim(
        theta_jn,
        phi_jl,
        tilt_1,
        tilt_2,
        phi_12,
        a_1,
        a_2,
        mass_1,
        mass_2,
        f_ref,
        phase,
    )

    params = dict(
        m1=mass_1.item() * MSUN,
        m2=mass_2.item() * MSUN,
        S1x=chi1x.item(),
        S1y=chi1y.item(),
        S1z=chi1z.item(),
        S2x=chi2x.item(),
        S2y=chi2y.item(),
        S2z=chi2z.item(),
        distance=(distance.item() * u.Mpc).to("m").value,
        inclination=inclination.item(),
        phiRef=phase.item(),
        longAscNodes=0.0,
        eccentricity=0.0,
        meanPerAno=0.0,
        deltaF=1.0 / sample_rate,
        f_min=10.0,
        f_ref=f_ref,
        f_max=300,
        approximant=lalsimulation.IMRPhenomPv2,
        LALpars=lal.CreateDict(),
    )
    hp_lal, _ = lalsimulation.SimInspiralChooseFDWaveform(**params)
    hp_lal = hp_lal.data.data.real

    slice_length = int(0.15 * sample_rate)

    for ftype, (rp, rs) in zip(filters, rprs):
        b, a = iirfilter(
            order,
            low_cutoff,
            btype="low",
            analog=False,
            output="ba",
            fs=sample_rate,
            rp=rp,
            rs=rs,
            ftype=ftype,
        )

        scipy_filtered_data_low = filtfilt(b, a, hp_lal)[
            slice_length:-slice_length
        ]

        b, a = iirfilter(
            order,
            high_cutoff,
            btype="high",
            analog=False,
            output="ba",
            fs=sample_rate,
            rp=rp,
            rs=rs,
            ftype=ftype,
        )
        scipy_filtered_data_high = filtfilt(b, a, hp_lal)[
            slice_length:-slice_length
        ]

        torch_filtered_data_low = IIRFilter(
            order,
            low_cutoff,
            btype="low",
            analog=False,
            fs=sample_rate,
            rs=rs,
            rp=rp,
            ftype=ftype,
        )(torch.tensor(hp_lal).repeat(10, 1))[
            :, slice_length:-slice_length
        ].numpy()

        torch_filtered_data_high = IIRFilter(
            order,
            high_cutoff,
            btype="high",
            analog=False,
            fs=sample_rate,
            rs=rs,
            rp=rp,
            ftype=ftype,
        )(torch.tensor(hp_lal).repeat(10, 1))[
            :, slice_length:-slice_length
        ].numpy()

        # test batch processing
        for i in range(9):
            assert np.allclose(
                torch_filtered_data_low[0],
                torch_filtered_data_low[i + 1],
                atol=float(np.finfo(float).eps),
            )
            assert np.allclose(
                torch_filtered_data_high[0],
                torch_filtered_data_high[i + 1],
                atol=float(np.finfo(float).eps),
            )

        assert np.allclose(
            1e21 * scipy_filtered_data_low,
            1e21 * torch_filtered_data_low[0],
            atol=7e-3,
        )
        assert np.allclose(
            1e21 * scipy_filtered_data_high,
            1e21 * torch_filtered_data_high[0],
            atol=7e-3,
        )
