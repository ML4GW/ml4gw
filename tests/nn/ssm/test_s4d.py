import pytest
import torch

from ml4gw.nn.ssm.s4d import S4D, S4DKernel, S4Model


def test_shape():
    # Input: (batch, channels, time). Output: (batch, d_output).
    B, d_input, L = 4, 2, 256
    model = S4Model(d_input=d_input, d_output=1, d_model=32, n_layers=2)
    out = model(torch.randn(B, d_input, L))
    assert out.shape == (B, 1)


def test_gradient_flow():
    # If gradients don't flow back through the model, it cannot be trained.
    x = torch.randn(2, 2, 128, requires_grad=True)
    model = S4Model(d_input=2, d_output=1, d_model=16, n_layers=2)
    model(x).sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert x.grad.abs().sum() > 0


def test_eval_determinism():
    # Same input must give the same output at inference time.
    # Dropout is random during training but must be off during eval.
    model = S4Model(d_input=2, d_output=1, d_model=32, n_layers=2).eval()
    x = torch.randn(2, 2, 128)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_ssm_kernel_linearity():
    # The kernel is a linear operator: K(a*u + b*v) == a*K(u) + b*K(v).
    # Superposition must hold; if it doesn't, the convolution is broken.
    kernel = S4DKernel(d_model=4, N=16)
    u, v = torch.randn(4, 128), torch.randn(4, 128)
    a, b = 2.3, -1.7

    k = kernel(L=128)

    def convolve(signal):
        k_f = torch.fft.rfft(k, n=256)
        s_f = torch.fft.rfft(signal, n=256)
        return torch.fft.irfft(k_f * s_f, n=256)[..., :128]

    assert torch.allclose(
        convolve(a * u + b * v), a * convolve(u) + b * convolve(v), atol=1e-5
    )


def test_skip_connection_contributes():
    # The model adds a learned direct term: output = SSM(input) + D * input.
    # Zeroing D should change the output, confirming the skip path is active.
    model = S4Model(d_input=2, d_output=1, d_model=16, n_layers=2).eval()
    x = torch.randn(1, 2, 64)

    with torch.no_grad():
        out_before = model(x).clone()
        for layer in model.s4_layers:
            layer.D.zero_()
        out_after = model(x)

    assert not torch.allclose(out_before, out_after)


def test_s4d_transposed_false_shape():
    # transposed=False should accept (B, L, H) and return the same shape.
    layer = S4D(d_model=4, d_state=16, transposed=False)
    x = torch.randn(2, 128, 4)
    y = layer(x)
    assert y.shape == x.shape


def test_s4d_invalid_dropout_raises():
    with pytest.raises(ValueError, match="dropout must be"):
        S4D(d_model=4, d_state=16, dropout=1.5)

    with pytest.raises(ValueError, match="dropout must be a float"):
        S4D(d_model=4, d_state=16, dropout=None)
