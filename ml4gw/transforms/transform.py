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

    def _load_from_state_dict(self, *args):
        # keep track of number of error messages to see
        # if trying to load these weights causes another
        num_errs = len(args[-1])
        ret = super()._load_from_state_dict(*args)

        # if we didn't create any new errors, then
        # assume that we built correctly
        if len(args[-1]) == num_errs:
            self.built = True
        return ret
