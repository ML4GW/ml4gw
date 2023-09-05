import h5py
import numpy as np
import pytest

from ml4gw.dataloading import Hdf5TimeSeriesDataset


class TestHdf5TimeSeriesDataset:
    @pytest.fixture
    def channels(self):
        return ["A", "B"]

    @pytest.fixture
    def sample_rate(self):
        return 128

    @pytest.fixture
    def kernel_size(self, sample_rate):
        return 2 * sample_rate

    @pytest.fixture
    def batch_size(self):
        return 128

    @pytest.fixture
    def batches_per_epoch(self):
        return 10

    @pytest.fixture
    def fnames(self, channels, sample_rate, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        fnames = {"a.h5": 10, "b.h5": 4, "c.h5": 6}
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

    @pytest.fixture(params=[True, False, "files"])
    def coincident(self, request):
        return request.param

    @pytest.fixture
    def dataset(
        self,
        fnames,
        channels,
        kernel_size,
        batch_size,
        batches_per_epoch,
        coincident,
    ):
        return Hdf5TimeSeriesDataset(
            sorted(fnames.keys()),
            channels,
            kernel_size,
            batch_size,
            batches_per_epoch,
            coincident=coincident,
        )

    def test_coincident_arg_value(
        self,
        fnames,
        channels,
        kernel_size,
        batch_size,
        batches_per_epoch,
    ):
        with pytest.raises(ValueError):
            Hdf5TimeSeriesDataset(
                sorted(fnames.keys()),
                channels,
                kernel_size,
                batch_size,
                batches_per_epoch,
                coincident="wrong",
            )

    def test_init(self, dataset):
        assert dataset.num_channels == 2
        assert len(dataset) == 10
        expected_probs = np.array([0.5, 0.2, 0.3])
        np.testing.assert_equal(expected_probs, dataset.probs)

    def test_sample_fnames(self, dataset):
        fnames = dataset.sample_fnames(size=(10,))
        assert len(fnames) == 10

        # really weak check: let's at least confirm
        # that we sample the 10s segment  more than
        # we sample the 4s segment.
        counts = {fname: 0 for fname in dataset.fnames}
        for _ in range(10):
            fnames = dataset.sample_fnames((10,))
            for fname in fnames:
                counts[fname] += 1
        assert counts["a.h5"] > counts["b.h5"]

    def test_sample_batch(self, dataset, kernel_size, coincident):
        x = dataset.sample_batch()
        assert x.shape == (128, 2, kernel_size)

        if coincident is True:
            # coincidence check is simple: they should all match
            for sample in x:
                assert (sample[0] == -sample[1]).all().item()
                assert (np.diff(sample[0].numpy()) == 1).all()
        else:
            # for non-coincident loading, do some checks around
            # which file each sampled channel corresponds to
            def get_bin(i):
                if i < (128 * 10):
                    return 0
                if i >= (128 * 14):
                    return 2
                return 1

            all_equal = True
            for sample in x:
                # check the files each channel came from
                bin0 = get_bin(sample[0, 0])
                bin1 = get_bin(-sample[1, 0])

                if bin0 != bin1:
                    # if they're not equal, we know that at
                    # least one batch element came out with
                    # non-coincident samples, which is good
                    # enough for us
                    all_equal = False

                    # if we're being fully non-coincident, then
                    # just the fact that we have any samples
                    # from different files means the check is done
                    if coincident is False:
                        break

                    # if we're sampling independently from within
                    # a given file, this is incorrect
                    raise ValueError(f"Got bins {bin0} and {bin1}")
                else:
                    # if the channels are from the same file,
                    # still check whether they come from the
                    # same part of the same file
                    all_equal &= (sample[0] == -sample[1]).all().item()
            else:
                # only get here if the loop never broke, which is only
                # a problem when we're sampling fully non-coincidentally
                if self.coincident is False:
                    raise ValueError("All channels came from same file")

            # if all channels came out the same, something went wrong
            if all_equal:
                raise ValueError("Sampling was coincident")

    def test_iter(self, dataset, kernel_size):
        # test_sample_batch covered most our checks, so here
        # we'll just make sure that we respect the dataset lengths
        for i, x in enumerate(dataset):
            assert x.shape == (128, 2, kernel_size)
        assert i == 9
