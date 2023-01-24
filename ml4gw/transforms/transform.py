from typing import Any, Mapping

import torch


class FittableTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.built = False

    def build(self, **params):
        state_dict = self.state_dict()
        for name, value in params.items():
            state_dict[name].copy_(value)
        self.built = True

    def _check_built(self):
        if not self.built:
            raise ValueError(
                "Must fit parameters of {} transform to data "
                "before calling forward step".format(self.__class__.__name__)
            )

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        keys = super().load_state_dict(state_dict, strict)
        self.built = True
        return keys
