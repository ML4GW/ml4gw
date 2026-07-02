import pytest
import torch

from ml4gw.utils.interferometer import InterferometerGeometry


@pytest.mark.parametrize("name", ["H1", "L1", "V1", "K1"])
def test_geometry_attributes_exist(name):
    geo = InterferometerGeometry(name)
    for attr in ("x_arm", "y_arm", "vertex"):
        assert isinstance(getattr(geo, attr), torch.Tensor)
        assert getattr(geo, attr).shape == (3,)


@pytest.mark.parametrize("name", ["H1", "L1", "V1", "K1"])
def test_arms_are_unit_vectors(name):
    geo = InterferometerGeometry(name)
    torch.testing.assert_close(
        torch.linalg.norm(geo.x_arm), torch.tensor(1.0), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        torch.linalg.norm(geo.y_arm), torch.tensor(1.0), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("name", ["E1", "GEO600", "h1", ""])
def test_invalid_name_raises(name):
    with pytest.raises(
        ValueError, match="is not recognized as an interferometer"
    ):
        InterferometerGeometry(name)


def test_h1_known_values():
    geo = InterferometerGeometry("H1")
    expected = torch.tensor([-0.22389266154, 0.79983062746, 0.55690487831])
    torch.testing.assert_close(geo.x_arm, expected, atol=1e-5, rtol=0)
