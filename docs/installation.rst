============
Installation
============

Pip
===
ml4gw is installable with pip:

.. code-block:: console
       
    $ pip install ml4gw


To build with a specific version of PyTorch/CUDA, please see the PyTorch installation `instructions <https://pytorch.org/get-started/previous-versions/>`_
to see how to specify the desired torch version and :code:`--extra-index-url` flag. For example, to install with PyTorch 2.5.0 and CUDA 12.1, use the following command:

.. code-block:: console

    $ pip install ml4gw torch==2.5.0 --extra-index-url=https://download.pytorch.org/whl/cu121

uv
==
If you want to develop ``ml4gw``, you can use `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ to install the project in editable mode.
For example, after cloning the repository, create a virtualenv using

.. code-block:: console

    $ uv venv --python=3.11

Then sync the dependencies from the uv lock file using

.. code-block:: console

    $ uv sync --all-extras

The suite of unit tests can be run using

.. code-block:: console

    $ uv run pytest
