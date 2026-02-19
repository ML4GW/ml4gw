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
        proj = FreqDomainSVDProjection(num_channels, n_freq, n_svd, V=V)
        expected = torch.from_numpy(V).float().transpose(0, 1)
        assert torch.allclose(proj.projection.weight.data, expected)

    def test_init_with_torch_V(self, num_channels, n_freq, n_svd):
        """Initialization with a torch V tensor sets weights."""
        V = torch.randn(2 * n_freq, n_svd)
        proj = FreqDomainSVDProjection(num_channels, n_freq, n_svd, V=V)
        expected = V.float().transpose(0, 1)
        assert torch.allclose(proj.projection.weight.data, expected)

    def test_per_channel_init_with_V(self, n_freq, n_svd):
        """Per-channel projections all start from the same V."""
        num_channels = 2
        V = torch.randn(2 * n_freq, n_svd)
        proj = FreqDomainSVDProjection(
            num_channels, n_freq, n_svd, V=V, per_channel=True
        )
        expected = V.float().transpose(0, 1)
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
        proj = FreqDomainSVDProjection(2, n_freq, n_svd, per_channel=True)
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


class TestComputeBasis:
    def test_output_shapes(self):
        """V and s have correct shapes."""
        n_waveforms, n_samples, n_svd = 200, 256, 20
        waveforms = np.random.randn(n_waveforms, n_samples).astype(
            np.float32
        )
        V, s = FreqDomainSVDProjection.compute_basis(waveforms, n_svd)
        n_freq = n_samples // 2 + 1
        assert V.shape == (2 * n_freq, n_svd)
        assert s.shape == (n_svd,)

    def test_singular_values_sorted(self):
        """Singular values are in descending order."""
        waveforms = np.random.randn(200, 256).astype(np.float32)
        _, s = FreqDomainSVDProjection.compute_basis(waveforms, n_svd=20)
        assert np.all(np.diff(s) <= 0)

    def test_V_columns_orthonormal(self):
        """V columns are approximately orthonormal."""
        waveforms = np.random.randn(500, 128).astype(np.float32)
        V, _ = FreqDomainSVDProjection.compute_basis(waveforms, n_svd=30)
        VtV = V.T @ V
        assert np.allclose(VtV, np.eye(30), atol=1e-5)

    def test_frequency_domain_input(self):
        """Frequency-domain input produces same result as time-domain."""
        waveforms_td = np.random.randn(200, 256).astype(np.float32)
        waveforms_fd = np.fft.rfft(waveforms_td, axis=-1)

        V_td, s_td = FreqDomainSVDProjection.compute_basis(
            waveforms_td, n_svd=20, domain="time"
        )
        V_fd, s_fd = FreqDomainSVDProjection.compute_basis(
            waveforms_fd, n_svd=20, domain="frequency"
        )
        assert np.allclose(s_td, s_fd, atol=1e-4)
        # V columns may differ in sign; compare absolute values
        assert np.allclose(np.abs(V_td), np.abs(V_fd), atol=1e-4)

    def test_torch_input(self):
        """Accepts torch Tensor input."""
        waveforms = torch.randn(200, 256)
        V, s = FreqDomainSVDProjection.compute_basis(waveforms, n_svd=20)
        n_freq = 256 // 2 + 1
        assert V.shape == (2 * n_freq, 20)
        assert isinstance(V, np.ndarray)

    def test_initializes_projection(self):
        """Computed V correctly initializes a projection layer."""
        n_samples = 256
        n_svd = 20
        n_freq = n_samples // 2 + 1
        waveforms = np.random.randn(200, n_samples).astype(np.float32)

        V, _ = FreqDomainSVDProjection.compute_basis(waveforms, n_svd)
        proj = FreqDomainSVDProjection(
            num_channels=2, n_freq=n_freq, n_svd=n_svd, V=V
        )

        # Verify forward pass works
        x = torch.randn(4, 2, n_samples)
        y = proj(x)
        assert y.shape == (4, 2 * n_svd)

    def test_reconstruction_quality(self):
        """More SVD components capture more signal energy."""
        # Create waveforms with low-rank structure
        n_waveforms, n_samples = 500, 256
        rng = np.random.default_rng(42)
        # Low-rank: 5 basis signals + noise
        basis = rng.standard_normal((5, n_samples))
        coeffs = rng.standard_normal((n_waveforms, 5))
        waveforms = (coeffs @ basis).astype(np.float32)

        V_few, _ = FreqDomainSVDProjection.compute_basis(
            waveforms, n_svd=2
        )
        V_many, _ = FreqDomainSVDProjection.compute_basis(
            waveforms, n_svd=20
        )

        # Project and reconstruct with each basis
        freq_data = np.fft.rfft(waveforms, axis=-1)
        data_ri = np.concatenate(
            [freq_data.real, freq_data.imag], axis=-1
        )

        recon_few = data_ri @ V_few @ V_few.T
        recon_many = data_ri @ V_many @ V_many.T

        err_few = np.linalg.norm(data_ri - recon_few)
        err_many = np.linalg.norm(data_ri - recon_many)
        assert err_many < err_few

    def test_invalid_domain_raises(self):
        """Invalid domain argument raises ValueError."""
        waveforms = np.random.randn(100, 128).astype(np.float32)
        with pytest.raises(ValueError, match="domain must be"):
            FreqDomainSVDProjection.compute_basis(
                waveforms, n_svd=10, domain="invalid"
            )

    def test_n_svd_clamped(self):
        """n_svd is clamped to min(matrix dims) - 1."""
        # 10 waveforms with 64 samples -> matrix is (10, 66)
        # min dim - 1 = 9
        waveforms = np.random.randn(10, 64).astype(np.float32)
        V, s = FreqDomainSVDProjection.compute_basis(
            waveforms, n_svd=100
        )
        assert V.shape[1] == 9
        assert s.shape[0] == 9

    def test_reproducible(self):
        """Same random_state produces identical results."""
        waveforms = np.random.randn(200, 128).astype(np.float32)
        V1, s1 = FreqDomainSVDProjection.compute_basis(
            waveforms, n_svd=20, random_state=123
        )
        V2, s2 = FreqDomainSVDProjection.compute_basis(
            waveforms, n_svd=20, random_state=123
        )
        assert np.allclose(V1, V2)
        assert np.allclose(s1, s2)
