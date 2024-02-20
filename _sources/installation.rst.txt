============
Installation
============

Pip
===
ml4gw is installable with pip:

.. code-block:: console
       
    $ pip install ml4gw


To build with a specific version of PyTorch/CUDA, please see the PyTorch installation `instructions <https://pytorch.org/get-started/previous-versions/>`_
to see how to specify the desired torch version and :code:`--extra-index-url` flag. For example, to install with PyTorch 1.12 and CUDA 11.6, use the following command:

.. code-block:: console
    
        $ pip install ml4gw torch==1.12.0 --extra-index-url=https://download.pytorch.org/whl/cu116

Poetry
======
:code:`ml4gw` is also fully compatible with `Poetry <https://python-poetry.org/>`_, with a :code:`pyproject.toml` set up like

.. code-block:: toml

    [tool.poetry.dependencies]
    python = "^3.8"  # python versions 3.8-3.10 are supported
    ml4gw = "^0.1.0"
    torch = {version = "^1.12", source = "torch"}

    [[tool.poetry.source]]
    name = "torch"
    url = "https://download.pytorch.org/whl/cu116"
    secondary = true
    default = false

which will install ml4gw with PyTorch 1.12 built with CUDA 11.6.
