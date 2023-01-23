# ML4GW

Torch utilities for training neural networks in gravitational wave physics applications.

## Installation
### Pip installation
If have [Poetry](https://python-poetry.org/) installed as a pip build backend, you can install `ml4gw` by pointing `pip` at this repo:

```console
pip install git+https://github.com/ML4GW/ml4gw.git
```

If you have a specific version of CUDA you need to build `torch` against, please see the PyTorch installation instructions [here](https://pytorch.org/) (specifically you can include a `--extra-index-url` kwarg pointing to the location of the desired `torch` wheels).

### Poetry installation
`ml4gw` is also fully compatible with use in Poetry, with your `pyproject.toml` set up like

```toml
[tool.poetry.dependencies]
python = "^3.8"  # python versions 3.8-3.10 are supported
ml4gw = {git = "https://github.com/ml4gw/ml4gw.git", branch = "main"}
```

If, for example, you need to build against CUDA 11.6, you can up your `pyproject.toml` to use a secondary source for `torch`

```toml
[tool.poetry.dependencies]
python = "^3.8"
ml4gw = {git = "https://github.com/ml4gw/ml4gw.git", branch = "main"}
torch = {version = "^1.12", source = "torch"}

[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cu116"
secondary = true
default = false
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
KERNEL_LENGTH = 4
DEVICE = "cuda"  # or "cpu", wherever you want to run

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

dummy_data = np.random.randn(NUM_IFOS, DATA_LENGTH * SAMPLE_RATE)

# this will create a dataloader that iterates through your
# timeseries data sampling 4s long windows of data randomly
# and non-coincidentally: i.e. the background from each IFO
# will be sampled independently
dataset = InMemoryDataset(
    dummy_data,
    kernel_size=KERNEL_LENGTH * SAMPLE_RATE,
    batch_size=BATCH_SIZE,
    batches_per_epoch=50,
    coincident=False,
    shuffle=True,
    device=DEVICE  # this will move your dataset to GPU up-front if "cuda"
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
).to(DEVICE)

optimizer = torch.optim.Adam(nn.parameters(), lr=LEARNING_RATE)

spectral_density = SpectralDensity(SAMPLE_RATE, fftlength=2).to(DEVICE)

def fd_mse(X, y):
    """
    MSE in frequency domain. Obviously this doesn't
    give you much on its own, but you can imagine doing
    something like masking to just the bins you care about.
    """
    X = spectral_density(X)
    y = spectral_density(y)
    return ((X - y)**2).mean()


for i in range(NUM_EPOCHS):
    epoch_loss = 0
    for X in dataset:
        optimizer.zero_grad(set_to_none=True)
        assert X.shape == (32, NUM_IFOS, KERNEL_LENGTH * SAMPLE_RATE)
        y = nn(X)

        loss = loss_function(X, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(dataset)
    print(f"Epoch {i + 1}/{NUM_EPOCHS} Loss: {epoch_loss:0.3e}")
```

## Development
As this library is still very much a work in progress, we anticipate that novel use cases will encounter errors stemming from a lack of robustness.
We encourage users who encounter these difficulties to file issues on GitHub, and we'll be happy to offer support to extend our coverage to new or improved functionality.
We also strongly encourage ML users in the GW physics space to try their hand at working on these issues and joining on as collaborators!
For more information about how to get involved, feel free to reach out to [alecg@mit.edu](mailto:alecg@mit.edu) .
By bringing in new users with new use cases, we hope to develop this library into a truly general-purpose tool which makes DL more accessible for gravitational wave physicists everywhere.
