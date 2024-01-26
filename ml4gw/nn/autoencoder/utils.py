import torch


def match_size(X: torch.Tensor, target_size: int):
    diff = target_size - X.size(-1)
    left = int(diff // 2)
    right = diff - left

    if diff > 0:
        return torch.nn.functional.pad(X, (left, right))
    elif diff < 0:
        right = -right or None
        return X[:, :, -left:right]
    return X
