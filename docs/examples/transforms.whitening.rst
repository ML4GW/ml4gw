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
depending on future samples. The dynamic transform accepts an ASD at call time:

.. code-block:: python

   from ml4gw.transforms import MinimumPhaseWhiten

   whitener = MinimumPhaseWhiten(
      fduration=2,
      sample_rate=2048,
      highpass=20,
   )
   X_whitened = whitener(X, asd)

For a fixed background, fit one ASD per channel once and reuse the stored
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
causal minimum-phase filter. A causal minimum-phase response has
frequency-dependent group delay, and changing the target magnitude with a
highpass or lowpass cutoff also changes that delay, particularly near the
transition frequencies. Cropping removes the initial filter warm-up but does
not compensate for this frequency-dependent delay.

By default, both transforms crop
``int(fduration * sample_rate)`` samples from the left edge. This matches the
output length of :class:`~ml4gw.transforms.Whiten` for the even filter lengths
used by the existing transforms, but the alignment is different: standard whitening removes
half the filter duration from each edge, whereas minimum-phase whitening
removes the full duration from the left. Outputs from the two transforms
therefore should not be treated as sample-aligned.

Pass ``crop=False`` to keep the input length and retain samples computed from
zero-valued initial history. For consecutive chunks, prepend
``int(fduration * sample_rate)`` real historical samples before calling the
transform. Directly supplied ASDs at physical strain scale should use double
precision to avoid underflow during filter construction. Both transforms
return ``torch.float32`` outputs, matching the existing whitening transforms.
The minimum-phase transforms do not subtract a mean over the complete input
segment, since that operation would depend on future samples.
