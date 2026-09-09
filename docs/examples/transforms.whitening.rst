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

Minimum-phase whitening
-----------------------

For online applications, a minimum-phase filter can whiten data without
depending on future samples. Fit the filter from one background timeseries or
PSD per channel, then apply it to tensors of any duration:

.. code-block:: python

   from ml4gw.transforms import MinimumPhaseWhiten

   whitener = MinimumPhaseWhiten(
      num_channels=2,
      kernel_length=2,
      sample_rate=2048,
   )

   # Passing fftlength means the inputs are interpreted as timeseries.
   whitener.fit(background_h1, background_l1, fftlength=2)
   X_whitened = whitener(X)

The output has the same shape as the input. The first
``kernel_length * sample_rate - 1`` samples use zero-valued history; when
processing consecutive chunks, prepend that many real historical samples and
discard their outputs.
