Spectral Density
================

.. code-block:: python

   """
   Compute the power spectral density and cross-spectral density of
   batches of time-series data.
   """

   from ml4gw.transforms import SpectralDensity
   import torch

   sample_rate = 2048
   fftlength = 2
   duration = 10

   # Initialize the spectral transform with a sample rate and fftlength
   spectral_density = SpectralDensity(
      sample_rate=sample_rate, 
      fftlength=fftlength,
   )

   # Example data with shape (batch_size, channels, length)
   X = torch.randn(10, 2, duration * sample_rate)

   # Apply the spectral transform to compute the power spectral density of X
   psd = spectral_density(X)

   # This module can also compute the cross-spectral density between two signals
   Y = torch.randn(10, 2, duration * sample_rate)
   csd = spectral_density(X, Y)
