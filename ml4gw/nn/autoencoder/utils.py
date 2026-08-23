import torch
from torch import Tensor


def match_size(X: Tensor, target_size: int) -> Tensor:
    diff = target_size - X.size(-1)
    if diff > 0:
        left = int(diff // 2)
        right = diff - left
        return torch.nn.functional.pad(X, (left, right))
    elif diff < 0:
        crop = -diff
        left = int(crop // 2)
        right = crop - left
        end = -right if right > 0 else None
        return X[..., left:end]
    return X
