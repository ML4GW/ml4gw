import h5py
import numpy as np
import pytest
import torch

from ml4gw.dataloading.chunked_dataset import ChunkedDataset, ChunkLoader


@pytest.fixture
def channels():
    return ["A", "B"]


@pytest.fixture
def sample_rate():
    return 128


@pytest.fixture
def fnames(channels, sample_rate, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    fnames = {"a.h5": 10, "b.h5": 4, "c.h5": 6}
    idx = 0
    for fname, length in fnames.items():
        with h5py.File(fname, "w") as f:
            size = int(length * sample_rate)
            x = np.arange(idx, idx + size)
            f[channels[0]] = x
            f[channels[1]] = -x
            idx += size
    return fnames


class TestChunkLoader:
    @pytest.fixture
    def chunk_length(self):
        return 2

    @pytest.fixture
    def reads_per_chunk(self):
        return 4

    @pytest.fixture
    def chunks_per_epoch(self):
        return 6

    @pytest.fixture
    def loader(
        self,
        fnames,
        channels,
        sample_rate,
        chunk_length,
        reads_per_chunk,
        chunks_per_epoch,
    ):
        return ChunkLoader(
            sorted(fnames.keys()),
            channels,
            int(chunk_length * sample_rate),
            reads_per_chunk,
            chunks_per_epoch,
        )

    def test_init(self, loader):
        assert loader.coincident
        expected_probs = np.array([0.5, 0.2, 0.3])
        np.testing.assert_equal(expected_probs, loader.probs)

    def test_sample_fnames(self, loader):
        fnames = loader.sample_fnames()
        assert len(fnames) == 4

        # really weak check: let's at least confirm
        # that we sample the 10s segment  more than
        # we sample the 4s segment.
        counts = {fname: 0 for fname in loader.fnames}
        for i in range(10):
            fnames = loader.sample_fnames()
            for fname in fnames:
                counts[fname] += 1
        assert counts["a.h5"] > counts["b.h5"]

    def test_load_coincident(self, loader, sample_rate):
        x = loader.load_coincident()
        assert x.shape == (4, 2, 2 * sample_rate)

        for sample in x:
            np.testing.assert_equal(sample[0], -sample[1])
            np.testing.assert_equal(np.diff(sample[0]), 1)

    def test_load_noncoincident(self, loader, sample_rate):
        x = loader.load_noncoincident()
        assert x.shape == (4, 2, 2 * sample_rate)

        for sample in x:
            assert (sample[0] != -sample[1]).all()
            np.testing.assert_equal(np.diff(sample[0]), 1)

    def test_iter(self, loader, sample_rate):
        for i, x in enumerate(loader):
            assert x.shape == (4, 2, 2 * sample_rate)

            for sample in x:
                torch.testing.assert_close(
                    sample[0], -sample[1], rtol=0, atol=0
                )
                diffs = torch.diff(sample[0])
                expected = torch.ones_like(diffs)
                torch.testing.assert_close(diffs, expected, rtol=0, atol=0)
        assert i == 5


class TestChunkedDataset:
    @pytest.fixture
    def chunk_length(self):
        return 2

    @pytest.fixture
    def reads_per_chunk(self):
        return 4

    @pytest.fixture
    def chunks_per_epoch(self):
        return 6

    @pytest.fixture
    def kernel_length(self):
        return 1.5

    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def batches_per_chunk(self):
        return 7

    @pytest.fixture(params=[0, 3])
    def num_workers(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def coincident(self, request):
        return request.param

    @pytest.fixture
    def dataset(
        self,
        fnames,
        channels,
        kernel_length,
        sample_rate,
        batch_size,
        chunk_length,
        reads_per_chunk,
        batches_per_chunk,
        chunks_per_epoch,
        num_workers,
        coincident,
    ):
        return ChunkedDataset(
            sorted(fnames),
            channels,
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            batch_size=batch_size,
            reads_per_chunk=reads_per_chunk,
            chunk_length=chunk_length,
            batches_per_chunk=batches_per_chunk,
            chunks_per_epoch=chunks_per_epoch,
            num_workers=num_workers,
            coincident=coincident,
            pin_memory=False,
        )

    def test_init(self, dataset, num_workers):
        assert dataset.chunk_size == 256
        assert dataset.kernel_size == 192
        if num_workers:
            assert dataset.chunk_loader.dataset.reads_per_chunk == 1

    def test_iter(self, dataset, coincident):
        for i, x in enumerate(dataset):
            assert x.shape == (8, 2, 192)
            if coincident:
                torch.testing.assert_close(x[:, 0], -x[:, 1], rtol=0, atol=0)
            else:
                assert (x[:, 0] != -x[:, 1]).any().item()

            diffs = torch.diff(x[:, 0], axis=-1)
            expected = torch.ones_like(diffs)
            torch.testing.assert_close(diffs, expected, rtol=0, atol=0)

        assert i == 41
