Whitening
=========

.. code-block:: python

   """
   Whiten a batch of time-series data
   """

   from ml4gw.transforms import Whiten
   import torch

   fduration = 2
   sample_rate = 2048
   highpass = 20
   duration = 10

   whitener = Whiten(
      fduration=fduration,
      sample_rate=sample_rate,
      highpass=highpass,
   )

   X = torch.randn(10, 2, duration * sample_rate)

   # Apply the whitening transform. Assume `psd` is the 
   # power spectral density computed using, e.g., the 
   # `SpectralDensity` transform.
   X_whitened = whitener(X, psd)
