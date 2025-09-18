# ML4GW
![PyPI - Version](https://img.shields.io/pypi/v/ml4gw)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ml4gw) 
![GitHub License](https://img.shields.io/github/license/ML4GW/ml4gw)
![Test status](https://github.com/ML4GW/ml4gw/actions/workflows/unit-tests.yaml/badge.svg)
![Coverage badge](https://raw.githubusercontent.com/ML4GW/ml4gw/python-coverage-comment-action-data/badge.svg)

Torch utilities for training neural networks in gravitational wave physics applications.

## Documentation
Please visit our [documentation page](https://ml4gw.github.io/ml4gw/) to see descriptions and examples of the functions and modules available in `ml4gw`.
We also have an interactive Jupyter notebook demonstrating much of the core functionality available [here](https://github.com/ML4GW/ml4gw/blob/main/docs/tutorials/ml4gw_tutorial.ipynb).
To run this notebook, download it from the above link and follow the instructions within it to install the required packages.
See also the [documentation page](https://ml4gw.github.io/ml4gw/tutorials/ml4gw_tutorial.html) for the tutorial to look
through it without running the code.

## Installation
### Pip installation
You can install `ml4gw` with pip:

```console
pip install ml4gw
```

To build with a specific version of PyTorch/CUDA, please see the PyTorch installation instructions [here](https://pytorch.org/get-started/previous-versions/) to see how to specify the desired torch version and `--extra-index-url` flag. For example, to install with torch 2.5.1 and CUDA 11.8 support, you would run

```console
pip install ml4gw torch==2.5.1--extra-index-url=https://download.pytorch.org/whl/cu118
```

### uv installation
If you want to develop `ml4gw`, you can use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the project in editable mode.
For example, after cloning the repository, create a virtualenv using
```bash
uv venv --python=3.11
```
Then sync the dependencies from the [uv lock file](/uv.lock) using
```bash
uv sync --all-extras
```
Code changes can be tested using
```bash
uv run pytest
```
See [contribution guide](/CONTRIBUTING.md) for more details.

## Contributing
If you come across errors in the code, have difficulties using this software, or simply find that the current version doesn't cover your use case, please file an issue on our GitHub page, and we'll be happy to offer support.
If you want to add feature, please refer to the [contribution guide](/CONTRIBUTING.md) for more details.
We also strongly encourage ML users in the GW physics space to try their hand at working on these issues and joining on as collaborators!
For more information about how to get involved, feel free to reach out to [ml4gw@ligo.mit.edu](mailto:ml4gw@ligo.mit.edu).
By bringing in new users with new use cases, we hope to develop this library into a truly general-purpose tool that makes deep learning more accessible for gravitational wave physicists everywhere.

## Funding
We are grateful for the support of the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for <a href="https://a3d3.ai">Accelerating AI Algorithms for Data Driven Discovery (A3D3)</a> under Cooperative Agreement No. <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997">PHY-2117997</a>.
