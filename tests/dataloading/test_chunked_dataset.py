import random

import pytest
import torch

from ml4gw.dataloading import ChunkedTimeSeriesDataset


class TestChunkedTimeseriesDataset:
    @pytest.fixture
    def chunk_length(self):
        return 2

    @pytest.fixture
    def chunks_per_epoch(self):
        return 6

    @pytest.fixture
    def chunk_it(self, chunk_length, chunks_per_epoch):
        def it():
            for i in range(chunks_per_epoch):
                chunk = []
                for _ in range(6):
                    start = random.randint(0, 128 * 8)
                    stop = start + 128 * chunk_length
                    row = torch.arange(start, stop)
                    row = torch.stack([row, -row])
                    chunk.append(row)
                chunk = torch.stack(chunk)
                yield chunk

        return it

    @pytest.fixture
    def kernel_length(self):
        return 1.5

    @pytest.fixture
    def batch_size(self):
        return 8

    @pytest.fixture
    def batches_per_chunk(self):
        return 7

    @pytest.fixture(params=[True, False])
    def coincident(self, request):
        return request.param

    @pytest.fixture
    def dataset(
        self,
        chunk_it,
        kernel_length,
        batch_size,
        batches_per_chunk,
        coincident,
    ):
        return ChunkedTimeSeriesDataset(
            chunk_it(),
            int(kernel_length * 128),
            batch_size=batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=coincident,
            device="cpu",
        )

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
