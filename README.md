# ML4GW
![PyPI - Version](https://img.shields.io/pypi/v/ml4gw)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ml4gw) 
![GitHub License](https://img.shields.io/github/license/ML4GW/ml4gw)
![Test status](https://github.com/ML4GW/ml4gw/actions/workflows/unit-tests.yaml/badge.svg)

Torch utilities for training neural networks in gravitational wave physics applications.

## Documentation
Please visit our [documentation page](https://ml4gw.github.io/ml4gw/) to see descriptions and examples of the functions and modules available in `ml4gw`.
We also have an interactive Jupyter notebook that demonstrates much of the core functionality available in the `examples` directory.

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

### Poetry installation
`ml4gw` is also fully compatible with use in Poetry, with your `pyproject.toml` set up like

```toml
[tool.poetry.dependencies]
python = "^3.9"  # python versions 3.9-3.12 are supported
ml4gw = "^0.6"
```

To build against a specific PyTorch/CUDA combination, consult the PyTorch installation documentation above and specify the `extra-index-url` via the `tool.poetry.source` table in your `pyproject.toml`. For example, to build against CUDA 11.6, you would do something like:

```toml
[tool.poetry.dependencies]
python = "^3.9"
ml4gw = "^0.6"
torch = {version = "^2.0", source = "torch"}

[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cu118"
priority = "explicit"
```

## Contributing
If you come across errors in the code, have difficulties using this software, or simply find that the current version doesn't cover your use case, please file an issue on our GitHub page, and we'll be happy to offer support.
We encourage users who encounter these difficulties to file issues on GitHub, and we'll be happy to offer support to extend our coverage to new or improved functionality.
We also strongly encourage ML users in the GW physics space to try their hand at working on these issues and joining on as collaborators!
For more information about how to get involved, feel free to reach out to [ml4gw@ligo.mit.edu](mailto:ml4gw@ligo.mit.edu).
By bringing in new users with new use cases, we hope to develop this library into a truly general-purpose tool that makes deep learning more accessible for gravitational wave physicists everywhere.

## Funding
We are grateful for the support of the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for <a href="https://a3d3.ai">Accelerating AI Algorithms for Data Driven Discovery (A3D3)</a> under Cooperative Agreement No. <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997">PHY-2117997</a>.
