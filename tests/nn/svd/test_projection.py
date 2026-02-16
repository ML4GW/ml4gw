import numpy as np
import pytest
import torch

from ml4gw.nn.svd.projection import FreqDomainSVDProjection


@pytest.fixture(params=[1, 2])
def num_channels(request):
    return request.param


@pytest.fixture(params=[65, 129])
def n_freq(request):
    return request.param


@pytest.fixture(params=[10, 50])
def n_svd(request):
    return request.param


@pytest.fixture(params=[True, False])
def per_channel(request):
    return request.param


@pytest.fixture(params=[4, 8])
def batch_size(request):
    return request.param


class TestFreqDomainSVDProjection:
    def test_output_shape(
        self, num_channels, n_freq, n_svd, per_channel, batch_size
    ):
        """Forward pass produces correct output shape."""
        proj = FreqDomainSVDProjection(
            num_channels, n_freq, n_svd, per_channel=per_channel
        )
        n_samples = (n_freq - 1) * 2
        x = torch.randn(batch_size, num_channels, n_samples)
        y = proj(x)
        assert y.shape == (batch_size, num_channels * n_svd)

    def test_output_dim_property(self, num_channels, n_freq, n_svd):
        """output_dim reports the correct value."""
        proj = FreqDomainSVDProjection(num_channels, n_freq, n_svd)
        assert proj.output_dim == num_channels * n_svd

    def test_init_with_numpy_V(self, num_channels, n_freq, n_svd):
        """Initialization with a numpy V matrix sets weights."""
        V = np.random.randn(2 * n_freq, n_svd).astype(np.float32)
        proj = FreqDomainSVDProjection(
            num_channels, n_freq, n_svd, V=V
        )
        expected = torch.from_numpy(V).float().T
        assert torch.allclose(proj.projection.weight.data, expected)

    def test_init_with_torch_V(self, num_channels, n_freq, n_svd):
        """Initialization with a torch V tensor sets weights."""
        V = torch.randn(2 * n_freq, n_svd)
        proj = FreqDomainSVDProjection(
            num_channels, n_freq, n_svd, V=V
        )
        expected = V.float().T
        assert torch.allclose(proj.projection.weight.data, expected)

    def test_per_channel_init_with_V(self, n_freq, n_svd):
        """Per-channel projections all start from the same V."""
        num_channels = 2
        V = torch.randn(2 * n_freq, n_svd)
        proj = FreqDomainSVDProjection(
            num_channels, n_freq, n_svd, V=V, per_channel=True
        )
        expected = V.float().T
        for ch_proj in proj.projections:
            assert torch.allclose(ch_proj.weight.data, expected)

    def test_V_shape_mismatch_raises(self):
        """Wrong V shape raises ValueError."""
        V = np.random.randn(100, 10).astype(np.float32)
        with pytest.raises(ValueError, match=r"V must have shape"):
            FreqDomainSVDProjection(2, 65, 10, V=V)

    def test_freeze_unfreeze_shared(self, n_freq, n_svd):
        """freeze/unfreeze controls requires_grad for shared."""
        proj = FreqDomainSVDProjection(2, n_freq, n_svd)
        proj.freeze()
        for p in proj.projection.parameters():
            assert not p.requires_grad
        proj.unfreeze()
        for p in proj.projection.parameters():
            assert p.requires_grad

    def test_freeze_unfreeze_per_channel(self, n_freq, n_svd):
        """freeze/unfreeze controls requires_grad for per-channel."""
        proj = FreqDomainSVDProjection(
            2, n_freq, n_svd, per_channel=True
        )
        proj.freeze()
        for ch_proj in proj.projections:
            for p in ch_proj.parameters():
                assert not p.requires_grad
        proj.unfreeze()
        for ch_proj in proj.projections:
            for p in ch_proj.parameters():
                assert p.requires_grad

    def test_gradient_flows(self, n_freq, n_svd):
        """Gradient flows through the projection."""
        proj = FreqDomainSVDProjection(2, n_freq, n_svd)
        n_samples = (n_freq - 1) * 2
        x = torch.randn(4, 2, n_samples, requires_grad=True)
        y = proj(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert proj.projection.weight.grad is not None

    def test_frozen_no_gradient(self, n_freq, n_svd):
        """Frozen projection accumulates no gradient."""
        proj = FreqDomainSVDProjection(2, n_freq, n_svd)
        proj.freeze()
        n_samples = (n_freq - 1) * 2
        x = torch.randn(4, 2, n_samples, requires_grad=True)
        y = proj(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert proj.projection.weight.grad is None

    def test_shared_vs_per_channel_initial_output(self, n_freq, n_svd):
        """Shared and per-channel give same output with same V."""
        V = torch.randn(2 * n_freq, n_svd)
        shared = FreqDomainSVDProjection(
            2, n_freq, n_svd, V=V, per_channel=False
        )
        per_ch = FreqDomainSVDProjection(
            2, n_freq, n_svd, V=V, per_channel=True
        )
        n_samples = (n_freq - 1) * 2
        x = torch.randn(4, 2, n_samples)
        y_shared = shared(x)
        y_per_ch = per_ch(x)
        assert torch.allclose(y_shared, y_per_ch, atol=1e-4)

    def test_deterministic(self, n_freq, n_svd):
        """Same input produces same output (no stochastic layers)."""
        proj = FreqDomainSVDProjection(2, n_freq, n_svd)
        proj.eval()
        n_samples = (n_freq - 1) * 2
        x = torch.randn(4, 2, n_samples)
        y1 = proj(x)
        y2 = proj(x)
        assert torch.allclose(y1, y2)

    def test_save_load(self, tmp_path, n_freq, n_svd):
        """State dict round-trips correctly."""
        V = torch.randn(2 * n_freq, n_svd)
        proj = FreqDomainSVDProjection(2, n_freq, n_svd, V=V)

        path = tmp_path / "proj.pt"
        torch.save(proj.state_dict(), path)

        proj2 = FreqDomainSVDProjection(2, n_freq, n_svd)
        proj2.load_state_dict(torch.load(path, weights_only=True))

        n_samples = (n_freq - 1) * 2
        x = torch.randn(4, 2, n_samples)
        assert torch.allclose(proj(x), proj2(x))
