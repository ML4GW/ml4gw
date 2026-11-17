import torch

from ml4gw.nn.streaming import StreamRunner


def test_step_groups_outputs_by_stream(make_whitener):
    class Model(torch.nn.Module):
        def forward(self, X):
            return torch.arange(len(X), dtype=X.dtype)[:, None]

    runner = StreamRunner(make_whitener(), Model())

    # batch_size * stride_size samples
    update = torch.randn(2, 2, 64)
    outputs = runner.step(update)
    torch.testing.assert_close(
        outputs,
        torch.tensor(
            [[[0], [1], [2], [3]], [[4], [5], [6], [7]]], dtype=update.dtype
        ),
        rtol=0,
        atol=0,
    )


def test_run_concatenates_each_tuple_output(make_whitener):
    class Model(torch.nn.Module):
        def forward(self, X):
            out = torch.arange(len(X), dtype=X.dtype)[:, None]
            return out, out * 10

    runner = StreamRunner(make_whitener(), Model())
    # 128 samples for two updates
    data = torch.randn(2, 2, 128)
    outputs = runner.run(data)
    expected = torch.tensor(
        [
            [[0], [1], [2], [3], [0], [1], [2], [3]],
            [[4], [5], [6], [7], [4], [5], [6], [7]],
        ],
        dtype=data.dtype,
    )
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected * 10, rtol=0, atol=0)


def test_tuple_transform_output_passed_as_separate_arguments(make_whitener):
    class Model(torch.nn.Module):
        def forward(self, X, Y):
            return torch.arange(len(X), dtype=X.dtype)[:, None]

    class Transform(torch.nn.Module):
        def forward(self, X):
            return X, X

    runner = StreamRunner(make_whitener(), Model(), Transform())
    update = torch.randn(2, 2, 64)
    outputs = runner.step(update)

    assert outputs.shape == (2, 4, 1)


def test_run_concatenates_batches_and_drops_padded_outputs(make_whitener):
    class Model(torch.nn.Module):
        def forward(self, X):
            return torch.arange(len(X), dtype=X.dtype)[:, None]

    runner = StreamRunner(make_whitener(), Model())
    # 160 samples for 2.5 updates -> 10 outputs per stream
    data = torch.randn(2, 2, 160)
    outputs = runner.run(data)
    expected = torch.tensor(
        [
            [[0], [1], [2], [3], [0], [1], [2], [3], [0], [1]],
            [[4], [5], [6], [7], [4], [5], [6], [7], [4], [5]],
        ],
        dtype=data.dtype,
    )
    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
