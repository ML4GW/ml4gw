import lal
import numpy as np
import pytest
import torch
from lalsimulation import (
    GenerateBandAndTimeLimitedWhiteNoiseBurst as lalWhiteNoiseBursts,
)

from ml4gw.waveforms import WhiteNoiseBurst


@pytest.fixture(params=[2048, 4096])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[2.0, 4.0])
def duration(request):
    return request.param


@pytest.fixture(params=[0.02, 0.1, 0.5, 2.0])
def time_envelope(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[55.0, 570.0, 1085.0, 1600.0])
def frequency(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[32.0, 704.0, 1376.0, 2048.0])
def bandwidth(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[0.0, 0.5, 1.0])
def eccentricity(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[0, torch.pi / 2, torch.pi])
def phase(request):
    return torch.tensor(request.param, dtype=torch.float64)


@pytest.fixture(params=[3.0e-40, 3.0e-37, 2.5e-34])
def int_hdot_squared(request):
    return torch.tensor(request.param, dtype=torch.float64)


def int_hdot_sq(h, dt):
    hdot = np.gradient(h, dt, axis=-1)
    return np.sum(hdot**2) * dt


# LAL's rng option
rng_list = [
    "taus2",
    "mt19937",
    "ranlux",
    "gfsr4",
]
torch.manual_seed(42)


def test_Band_and_Time_Limited_White_Noise_Burst(
    sample_rate,
    duration,
    time_envelope,
    frequency,
    bandwidth,
    eccentricity,
    phase,
    int_hdot_squared,
):

    nyquist = sample_rate / 2
    if frequency > nyquist:
        pytest.skip("Peak frequeny is lager then nyquist.")

    # ML4GW
    wnb = WhiteNoiseBurst(sample_rate=sample_rate, duration=duration)

    duplication = 1
    if (time_envelope < 0.5) or (bandwidth <= 500):
        duplication = 10
    if (time_envelope < 0.1) or (bandwidth <= 100):
        duplication = 50
    if (time_envelope < 0.1) and (bandwidth < 100):
        duplication = 100
    cross, plus = wnb(
        time_envelope=time_envelope.unsqueeze(0).expand(duplication, -1),
        frequency=frequency.unsqueeze(0).expand(duplication, -1),
        bandwidth=bandwidth.unsqueeze(0).expand(duplication, -1),
        eccentricity=eccentricity.unsqueeze(0).expand(duplication, -1),
        phase=phase.unsqueeze(0).expand(duplication, -1),
        int_hdot_squared=int_hdot_squared.unsqueeze(0).expand(duplication, -1),
    )
    cross, plus = cross.numpy(), plus.numpy()
    ml4gw_samples = len(cross[0])

    # Data type conversion
    time_envelope = time_envelope.item()
    frequency = frequency.item()
    bandwidth = bandwidth.item()
    eccentricity = eccentricity.item()
    phase = phase.item()
    int_hdot_squared = int_hdot_squared.item()

    # LAL
    hplus = []
    hcross = []
    # hplus_ = []
    # hcross_ = []
    for seed in range(duplication):
        h_plus, h_cross = lalWhiteNoiseBursts(
            duration=time_envelope,
            frequency=frequency,
            bandwidth=bandwidth,
            eccentricity=eccentricity,
            phase=phase,
            int_hdot_squared=int_hdot_squared,
            delta_t=1 / sample_rate,
            rng=lal.gsl_rng("mt19937", seed),
        )
        hcross.append(h_cross.data.data)
        hplus.append(h_plus.data.data)

    hcross = np.vstack(hcross)
    hplus = np.vstack(hplus)
    lal_samples = len(hcross[0])

    if lal_samples < ml4gw_samples:
        start, stop = (
            ml4gw_samples // 2 - lal_samples // 2,
            ml4gw_samples // 2 + lal_samples // 2 + 1,
        )
        cross, plus = cross[:, start:stop], plus[:, start:stop]
        n_sample = lal_samples
    else:
        start, stop = (
            lal_samples // 2 - ml4gw_samples // 2,
            lal_samples // 2 + ml4gw_samples // 2,
        )
        hcross, hplus = hcross[:, start:stop], hplus[:, start:stop]
        n_sample = ml4gw_samples

    # --------------------
    # Test section
    # --------------------

    # 1. Time domain check (amp_mean, amp_std)
    cross_mean = np.abs(np.mean(cross)) / duplication
    hcross_mean = np.abs(np.mean(hcross)) / duplication

    plus_mean = np.abs(np.mean(plus)) / duplication
    hplus_mean = np.abs(np.mean(hplus)) / duplication

    cross_std = np.std(cross, axis=1).mean()
    hcross_std = np.std(hcross, axis=1).mean()
    cross_scale = (cross_std + hcross_std) / 2
    if cross_scale != 0:  # When eccentricity = 1 cross become 0
        cross_std /= cross_scale
        hcross_std /= cross_scale

    plus_std = np.std(plus, axis=1).mean()
    hplus_std = np.std(hplus, axis=1).mean()
    plus_scale = (plus_std + hplus_std) / 2
    plus_std /= plus_scale
    hplus_std /= plus_scale

    # We generate plus and cross polarizations using LAL
    # with identical parameters but different random seeds,
    # and compare the resulting waveforms to determine
    # the appropriate tolerance values.
    mean_rtol = 50
    std_atol, std_rtol = 0.15, 0.15
    assert np.isclose(cross_mean, hcross_mean, rtol=mean_rtol)
    assert np.isclose(plus_mean, hplus_mean, rtol=mean_rtol)

    assert np.isclose(cross_std, hcross_std, atol=std_atol, rtol=std_rtol)
    assert np.isclose(plus_std, hplus_std, atol=std_atol, rtol=std_rtol)

    # # 2. Frequency domain check (f_0, delta_f)
    freqs = np.fft.fftfreq(n_sample, 1 / sample_rate)
    freqs_shifted = np.fft.fftshift(freqs)

    fft_cross = np.abs(np.fft.fftshift(np.fft.fft(cross)))
    fft_hcross = np.abs(np.fft.fftshift(np.fft.fft(hcross)))

    fft_plus = np.abs(np.fft.fftshift(np.fft.fft(plus)))
    fft_hplus = np.abs(np.fft.fftshift(np.fft.fft(hplus)))

    fft_cross = np.average(fft_cross, axis=0)
    fft_hcross = np.average(fft_hcross, axis=0)
    fft_plus = np.average(fft_plus, axis=0)
    fft_hplus = np.average(fft_hplus, axis=0)

    # Theoretical frequency window
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_burst__h.html#ga54d4218466cf78c0495789c9f27e153d
    amp_cross = (np.max(fft_cross + fft_hcross)) / 2
    amp_plus = (np.max(fft_plus + fft_hplus)) / 2
    exponent = -1 / 2 * (freqs_shifted - frequency) ** 2 / (bandwidth / 2) ** 2
    gauss = np.exp(exponent)
    # Scale the theoretical window function to
    # the average amplitude of the two waveform.
    freq_envelope_cross = amp_cross * gauss
    freq_envelope_plus = amp_plus * gauss

    # Perform cross product to frequency window and comapre the difference.
    cross_fft_match = np.matmul(freq_envelope_cross, fft_cross)
    hcross_fft_match = np.matmul(freq_envelope_cross, fft_hcross)
    plus_fft_match = np.matmul(freq_envelope_plus, fft_plus)
    hplus_fft_match = np.matmul(freq_envelope_plus, fft_hplus)

    rtol = 0.20
    assert np.isclose(cross_fft_match, hcross_fft_match, atol=0, rtol=rtol)
    assert np.isclose(plus_fft_match, hplus_fft_match, atol=0, rtol=rtol)

    # 3. Energy level check (\int h_dot^2 dt)
    dt = 1.0 / sample_rate

    ml4gw_energy = int_hdot_sq(cross, dt) + int_hdot_sq(plus, dt)
    lal_energy = int_hdot_sq(hcross, dt) + int_hdot_sq(hplus, dt)

    assert np.isclose(
        ml4gw_energy, lal_energy, atol=duplication * int_hdot_squared / 10
    )