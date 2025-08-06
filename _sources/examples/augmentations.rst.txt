Augmentations
=============

Apply random time-series reversal and inversion augmentations to batches of data.

.. code-block:: python

   from ml4gw.augmentations import SignalInverter, SignalReverser
   import torch

   # Initialize augmentors with probability of applying the transformation
   inverter = SignalInverter(prob=0.25)
   reverser = SignalReverser(prob=0.5)

   # Example data with shape (batch_size, channels, length)
   X = torch.randn(10, 2, 1000)

   # Apply augmentations
   X = inverter(X)
   X = reverser(X)
