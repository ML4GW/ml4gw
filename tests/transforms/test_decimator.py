import torch
import pytest
import numpy as np
from scipy.interpolate import interp1d
from ml4gw.waveforms import IMRPhenomD
from ml4gw.transforms.decimator import Decimator
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components


def test_decimator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Waveform generation
    param_dict = {
        "chirp_mass": torch.tensor([1.4]),
        "mass_ratio": torch.tensor([0.8]),
        "chi1": torch.tensor([0.0]),
        "chi2": torch.tensor([0.0]),
        "distance": torch.tensor([100.0]),
        "phic": torch.tensor([0.0]),
        "inclination": torch.tensor([1.0]),
    }

    sample_rate = 2048
    waveform_duration = 60
    f_min = 20
    f_ref = 20

    approximant = IMRPhenomD().to(device)
    waveform_generator = TimeDomainCBCWaveformGenerator(
        approximant=approximant,
        sample_rate=sample_rate,
        f_min=f_min,
        duration=waveform_duration,
        right_pad=0.5,
        f_ref=f_ref,
    ).to(device)

    param_dict["mass_1"], param_dict["mass_2"] = (
        chirp_mass_and_mass_ratio_to_components(
            param_dict["chirp_mass"], param_dict["mass_ratio"]
        )
    )
    param_dict["s1z"], param_dict["s2z"] = (
        param_dict["chi1"],
        param_dict["chi2"],
    )

    hc, hp = waveform_generator(**param_dict)

    # Positive Tests
    # Decimation
    schedule = torch.tensor(
        [[0, 40, 256], [40, 58, 512], [58, 60, 2048]],
        dtype=torch.int,
        device=device,
    )

    decimator = Decimator(sample_rate=sample_rate, schedule=schedule)

    indices = decimator.idx

    dec_hc = decimator(hc)
    dec_hp = decimator(hp)

    time = torch.arange(0, waveform_duration, 1 / sample_rate).to(device)
    time_dec = time[indices]

    f_hc = interp1d(time_dec.cpu().numpy(), dec_hc.cpu().numpy(), kind="cubic")
    f_hp = interp1d(time_dec.cpu().numpy(), dec_hp.cpu().numpy(), kind="cubic")

    up_hc = f_hc(time.cpu().numpy())
    up_hp = f_hp(time.cpu().numpy())

    assert np.allclose(hc.cpu().numpy() * 1e22, up_hc * 1e22, atol=1e-2)
    assert np.allclose(hp.cpu().numpy() * 1e22, up_hp * 1e22, atol=1e-2)

    # Testing the split functionality
    # Each segment length matches schedule * target rate
    seg = decimator(hc, split=True)
    exp_len = (
        ((schedule[:, 1] - schedule[:, 0]) * schedule[:, 2]).long().tolist()
    )
    actual_len = [s.shape[-1] for s in seg]
    assert actual_len == exp_len

    # Negative Tests
    wrong_schedule_shape = torch.tensor([[0, 40]])
    with pytest.raises(ValueError, match="Schedule must be of shape"):
        Decimator(sample_rate=2048, schedule=wrong_schedule_shape)

    wrong_schedule_time = torch.tensor([[10, 5, 256]])
    with pytest.raises(ValueError, match="end_time > start_time"):
        Decimator(sample_rate=2048, schedule=wrong_schedule_time)

    wrong_target_sr = torch.tensor([[0, 10, 500]])
    with pytest.raises(ValueError, match="must be divisible"):
        Decimator(sample_rate=2048, schedule=wrong_target_sr)

    wrong_input_len = torch.randn(1, 1, 100)
    decimator_valid = Decimator(
        sample_rate=2048, schedule=torch.tensor([[0, 2, 256]])
    )
    with pytest.raises(ValueError, match="expected schedule duration"):
        decimator_valid(wrong_input_len)
