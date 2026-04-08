import pytest
import torch

from ml4gw.nn.svd.dense import DenseResidualBlock


@pytest.fixture(params=[32, 64, 128])
def dim(request):
    return request.param


@pytest.fixture(params=[0.0, 0.1, 0.5])
def dropout(request):
    return request.param


@pytest.fixture(params=[4, 16])
def batch_size(request):
    return request.param


class TestDenseResidualBlock:
    def test_output_shape(self, dim, dropout, batch_size):
        """Output has same shape as input."""
        block = DenseResidualBlock(dim, dropout)
        x = torch.randn(batch_size, dim)
        y = block(x)
        assert y.shape == (batch_size, dim)

    def test_residual_connection(self, dim):
        """Zero-initialized MLP gives output close to norm(x)."""
        block = DenseResidualBlock(dim, dropout=0.0)
        # Zero out the MLP weights so net(x) ≈ 0
        with torch.no_grad():
            for layer in block.net:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias"):
                    layer.bias.zero_()

        block.eval()
        x = torch.randn(8, dim)
        y = block(x)
        # With net(x)=0, output should be LayerNorm(x + 0) = LayerNorm(x)
        expected = block.norm(x)
        assert torch.allclose(y, expected, atol=1e-6)

    def test_gradient_flows(self, dim):
        """Gradient flows through the block."""
        block = DenseResidualBlock(dim)
        x = torch.randn(4, dim, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_train_eval_consistency(self, dim):
        """Output distribution is similar in train and eval modes.

        This is the key property that LayerNorm provides over
        BatchNorm — no train/eval discrepancy.
        """
        block = DenseResidualBlock(dim, dropout=0.0)

        x = torch.randn(32, dim)

        block.train()
        y_train = block(x)

        block.eval()
        y_eval = block(x)

        # With dropout=0, train and eval should be identical
        assert torch.allclose(y_train, y_eval, atol=1e-6)

    def test_uses_layer_norm(self, dim):
        """Block uses LayerNorm, not BatchNorm."""
        block = DenseResidualBlock(dim)
        assert isinstance(block.norm, torch.nn.LayerNorm)

    def test_deterministic_eval(self, dim):
        """Same input gives same output in eval mode."""
        block = DenseResidualBlock(dim, dropout=0.5)
        block.eval()
        x = torch.randn(8, dim)
        y1 = block(x)
        y2 = block(x)
        assert torch.allclose(y1, y2)

    def test_stacking(self, dim, dropout):
        """Multiple blocks can be stacked in a Sequential."""
        blocks = torch.nn.Sequential(
            DenseResidualBlock(dim, dropout),
            DenseResidualBlock(dim, dropout),
            DenseResidualBlock(dim, dropout),
        )
        x = torch.randn(8, dim)
        y = blocks(x)
        assert y.shape == (8, dim)

    def test_save_load(self, tmp_path, dim):
        """State dict round-trips correctly."""
        block = DenseResidualBlock(dim)
        path = tmp_path / "block.pt"
        torch.save(block.state_dict(), path)

        block2 = DenseResidualBlock(dim)
        block2.load_state_dict(torch.load(path, weights_only=True))

        block.eval()
        block2.eval()
        x = torch.randn(4, dim)
        assert torch.allclose(block(x), block2(x))
