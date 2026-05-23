"""Benchmarks for TimeDomainCBCWaveformGenerator."""

from constants import SAMPLE_RATE

from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

DURATION = 4.0
F_MIN = 20.0
F_REF = 20.0
RIGHT_PAD = 0.5


def test_td_generator_forward(benchmark, cbc_inputs, device):
    approximant = IMRPhenomD().to(device)
    model = TimeDomainCBCWaveformGenerator(
        approximant=approximant,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        f_min=F_MIN,
        f_ref=F_REF,
        right_pad=RIGHT_PAD,
    ).to(device)

    chirp_mass = cbc_inputs["chirp_mass"]
    mass_ratio = cbc_inputs["mass_ratio"]
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
        chirp_mass, mass_ratio
    )
    chi1 = cbc_inputs["chi1"]
    chi2 = cbc_inputs["chi2"]

    benchmark(
        model,
        chirp_mass=chirp_mass,
        mass_ratio=mass_ratio,
        mass_1=mass_1,
        mass_2=mass_2,
        s1z=chi1,
        s2z=chi2,
        chi1=chi1,
        chi2=chi2,
        distance=cbc_inputs["distance"],
        phic=cbc_inputs["phic"],
        inclination=cbc_inputs["inclination"],
    )
