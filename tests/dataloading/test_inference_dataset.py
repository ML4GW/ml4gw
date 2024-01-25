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

    @pytest.fixture(params=[None, [0, 1]])
    def shift_sizes(self, request):
        return request.param

    @pytest.fixture
    def fname(self, channels, sample_rate, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        fname = "a.h5"
        length = 1
        with h5py.File(fname, "w") as f:
            size = int(length * sample_rate)
            x = np.arange(size)
            f[channels[0]] = x
            f[channels[1]] = -x
        return fname

    @pytest.fixture
    def dataset(
        self,
        fname,
        channels,
        stride_size,
    ):
        def fn(shift_sizes):
            return InferenceDataset(
                fname,
                channels,
                stride_size,
                shift_sizes=shift_sizes,
            )

        return fn

    def test_init(self, dataset):
        dataset = dataset(None)
        assert dataset.size == 100
        assert len(dataset) == 10

    def test_iter(self, dataset, stride_size):
        dataset = dataset(None)
        first = np.arange(stride_size)
        for i, x in enumerate(dataset):
            assert np.all(x[0] == first)
            assert np.all(x[1] == -first)
            first += stride_size

        assert i == len(dataset) - 1

    def test_iter_with_shifts(self, dataset, stride_size):
        dataset = dataset([0, 1])
        assert dataset.size == 99
        assert len(dataset) == 10

        first = np.arange(stride_size)
        for i, x in enumerate(dataset):
            # account for one sample smaller stride yielded due
            # to shifting data by one smaple
            if i == len(dataset) - 1:
                assert np.all(x[0] == first[:-1])
                assert np.all(x[1] == -first[:-1] - 1)

            else:
                assert np.all(x[0] == first + 0)
                assert np.all(x[1] == -first - 1)
            first += stride_size
