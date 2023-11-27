import h5py
import numpy as np
import pytest

from ml4gw.dataloading import InferenceDataset


class TestInferenceDataset:
    @pytest.fixture
    def channels(self):
        return ["A", "B"]

    @pytest.fixture
    def sample_rate(self):
        return 100

    @pytest.fixture
    def stride_size(self):
        return 10

    @pytest.fixture
    def fnames(self, channels, sample_rate, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        fnames = {"a.h5": 1, "b.h5": 2, "c.h5": 3}
        idx = 0
        keys = sorted(fnames)
        for fname in keys:
            length = fnames[fname]
            with h5py.File(fname, "w") as f:
                size = int(length * sample_rate)
                x = np.arange(idx, idx + size)
                f[channels[0]] = x
                f[channels[1]] = -x
                idx += size
        return fnames

    @pytest.fixture
    def dataset(
        self,
        fnames,
        channels,
        stride_size,
    ):

        return InferenceDataset(
            sorted(fnames.keys()),
            channels,
            stride_size,
        )

    def test_init(self, dataset):
        assert dataset.sizes == {"a.h5": 100, "b.h5": 200, "c.h5": 300}
        assert len(dataset) == 60

    def test_iter(self, dataset, stride_size):
        first = np.arange(stride_size)
        for i, x in enumerate(dataset):
            assert np.all(x[0] == first)
            assert np.all(x[1] == -first)
            first += stride_size
