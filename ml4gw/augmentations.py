import torch


class SignalInverter(torch.nn.Module):
    """
    Takes a tensor of timeseries of arbitrary dimension
    and randomly inverts (i.e. h(t) -> -h(t))
    each timeseries with probability `prob`.

    Args:
        prob:
            Probability that a timeseries is inverted
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        mask = torch.rand(size=X.shape[:-1]) < self.prob
        X[mask] *= -1
        return X


class SignalReverser(torch.nn.Module):
    """
    Takes a tensor of timeseries of arbitrary dimension
    and randomly reverses (i.e. h(t) -> h(-t))
    each timeseries with probability `prob`.

    Args:
        prob:
            Probability that a kernel is reversed
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        mask = torch.rand(size=X.shape[:-1]) < self.prob
        X[mask] = X[mask].flip(-1)
        return X
