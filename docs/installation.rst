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
