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
depending on future samples. The dynamic transform accepts a PSD at call time:

.. code-block:: python

   from ml4gw.transforms import MinimumPhaseWhiten

   whitener = MinimumPhaseWhiten(
      fduration=2,
      sample_rate=2048,
      highpass=20,
   )
   X_whitened = whitener(X, psd)

For a fixed background, fit one PSD per channel once and reuse the stored
filter:

.. code-block:: python

   from ml4gw.transforms import FixedMinimumPhaseWhiten

   whitener = FixedMinimumPhaseWhiten(
      num_channels=2,
      fduration=2,
      sample_rate=2048,
   )

   # Passing fftlength means the inputs are interpreted as timeseries.
   whitener.fit(
      background_h1,
      background_l1,
      fftlength=2,
      highpass=20,
   )
   X_whitened = whitener(X)

The highpass and lowpass responses use the same inverse-spectrum truncation
as the standard whitening transforms before the response is converted to a
causal minimum-phase filter.

By default, both transforms crop
``int(fduration * sample_rate) - 1`` warm-up samples from the left edge. Pass
``crop=False`` to keep the input length and retain samples computed from
zero-valued initial history. For consecutive chunks, prepend that many real
historical samples before calling the transform. Directly supplied PSDs at
physical strain scale should use double precision to avoid underflow in
``torch.float32``. The minimum-phase transforms do not subtract a mean over
the complete input segment, since that operation would depend on future
samples.
