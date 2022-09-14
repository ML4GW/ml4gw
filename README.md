# ML4GW

Torch utilities for training neural networks in gravitational wave physics applications.

## Installation
At present, the only way to install this library is via [Poetry](https://python-poetry.org/). Start by cloning this repository (or adding as a submodule to your repository), then adding it as a local dependency in your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.8"  # python versions 3.8-3.10 are supported
ml4gw = {path = "path/to/ml4gw", develop = true}
```

You can then update your lockfile/environment via

```console
poetry update
```

## Use cases
This library provided utilities for both data iteration and transformation via dataloaders defined in `ml4gw/dataloading` and transform layers exposed in `ml4gw/transforms`. Lower level functions and utilies are defined at the top level of the library and in the `utils` library.

For example, to train a simple autoencoder using a cost function in frequency space, you might do something like:

```python
import numpy as np
import torch
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import SpectralDensity

SAMPLE_RATE = 2048
NUM_IFOS = 2
DATA_LENGTH = 128

dummy_data = np.random.randn(NUM_IFOS, DATA_LENGTH * SAMPLE_RATE)

# this will create a dataloader that iterates through your
# timeseries data sampling 1s long windows of data randomly
# and non-coincidentally: i.e. the background from each IFO
# will be sampled independently
dataset = InMemoryDataset(
    dummy_data,
    kernel_size=1 * SAMPLE_RATE,
    batch_size=32,
    batches_per_epoch=50,
    coincident=False,
    shuffle=True,
    device="cuda"  # this will move your dataset to GPU up-front
)


nn = torch.nn.Sequential(
    torch.nn.Conv1d(
        in_channels=2,
        out_channels=8,
        kernel_size=7
    ),
    torch.nn.ConvTranspose1d(
        in_channels=8,
        out_channels=2,
        kernel_size=7
    )
)

optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)

spectral_density = SpectralDensity(SAMPLE_RATE, fflength=2)

def loss_function(X_psd, y_psd):
    # insert logic here about how you want to optimize your nn
    # here's a simple MSE (which you obviously could have also
    # done in time space)
    return ((X_psd - y_psd)**2).mean()

for X in dataset:
    optimizer.zero_grad(set_to_none=True)
    assert X.shape == (32, NUM_IFOS, SAMPLE_RATE)
    y = nn(X)

    X_psd = spectral_density(X)
    y_psd = spectral_density(y)

    loss = loss_function(X_psd, y_psd)
    loss.backward()
    optimizer.step()
```

## Development
As this is a library which is still very much a work in progress, we anticipate that new users with new use cases will anticipate errors stemming from a lack of robustness.
We encourage these users to file issues on GitHub and, if they have the inclincation, to try their hand at fixing themselves.
Only by trying it on these new and exciting use cases will this develop into a truly general-purpose tool making DL more accessible for gravitational wave physicists everywhere.
