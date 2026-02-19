Frequency-domain SVD projection
================================

The :class:`~ml4gw.nn.svd.projection.FreqDomainSVDProjection` layer projects
multi-channel time-domain data (e.g. interferometer strain) onto a reduced
SVD basis in the frequency domain. This filters out noise components
orthogonal to the signal manifold, acting as a learned dimensionality
reduction layer.

The approach follows `DINGO <https://github.com/dingo-gw/dingo>`_
(Dax et al., "Real-Time Gravitational Wave Science with Neural Posterior
Estimation"). The SVD is computed on intrinsic waveform polarizations
(h+, hx) whitened by the detector ASD. Because any detector response
``h = F+ h+ + Fx hx`` is a linear combination of the two polarizations,
a basis spanning {h+, hx} works for any sky location and detector.

Computing an SVD basis from waveforms
-------------------------------------

Use :meth:`~ml4gw.nn.svd.projection.FreqDomainSVDProjection.compute_basis`
to build the right singular vectors V from a bank of waveforms.
The waveform bank can be generated using ml4gw's
:class:`~ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
with an approximant like
:class:`~ml4gw.waveforms.cbc.phenom_d.IMRPhenomD`:

.. code-block:: python

   import torch
   import numpy as np
   from ml4gw.nn.svd import FreqDomainSVDProjection
   from ml4gw.waveforms import IMRPhenomD
   from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

   # Set up a waveform generator
   sample_rate = 2048.0
   duration = 4.0
   approximant = IMRPhenomD()
   generator = TimeDomainCBCWaveformGenerator(
       approximant=approximant,
       sample_rate=sample_rate,
       duration=duration,
       f_min=20.0,
       f_ref=20.0,
       right_pad=0.5,
   )

   # Generate a bank of waveforms by sampling parameters
   # (use more waveforms in practice, e.g. 5000+)
   n_waveforms = 200
   hc, hp = generator(
       chirp_mass=torch.distributions.Uniform(1.0, 2.0).sample((n_waveforms,)),
       mass_ratio=torch.distributions.Uniform(1.0, 4.0).sample((n_waveforms,)),
       chi1=torch.zeros(n_waveforms),
       chi2=torch.zeros(n_waveforms),
       distance=torch.full((n_waveforms,), 100.0),  # Mpc
       phic=torch.zeros(n_waveforms),
       inclination=torch.zeros(n_waveforms),
       mass_1=torch.distributions.Uniform(1.0, 3.0).sample((n_waveforms,)),
       mass_2=torch.distributions.Uniform(1.0, 3.0).sample((n_waveforms,)),
       s1z=torch.zeros(n_waveforms),
       s2z=torch.zeros(n_waveforms),
   )

   # Use one polarization (or both) as the waveform bank
   # Shape: (n_waveforms, n_samples)
   waveforms = hp.float().numpy()
   n_samples = waveforms.shape[-1]

   # Compute SVD basis (time-domain input is FFT'd internally)
   n_svd = 50
   V, s = FreqDomainSVDProjection.compute_basis(waveforms, n_svd=n_svd)

   # V: (2 * n_freq, n_svd) — right singular vectors
   # s: (n_svd,) — singular values (descending order)
   n_freq = n_samples // 2 + 1
   print(f"V shape: {V.shape}")   # (2 * n_freq, 50)
   print(f"s shape: {s.shape}")   # (50,)

   # Check cumulative energy capture
   total_variance = np.sum(s**2)
   cum_variance = np.cumsum(s**2) / total_variance
   print(f"Energy captured by {n_svd} components: {cum_variance[-1]:.4f}")

If your waveforms are already in the frequency domain (complex-valued),
pass ``domain="frequency"``:

.. code-block:: python

   # Pre-computed frequency-domain waveforms
   freq_waveforms = np.fft.rfft(waveforms, axis=-1)
   V_fd, s_fd = FreqDomainSVDProjection.compute_basis(
       freq_waveforms, n_svd=50, domain="frequency"
   )

Initializing a projection layer
--------------------------------

Pass the computed V matrix to initialize the projection weights:

.. code-block:: python

   import torch
   from ml4gw.nn.svd import FreqDomainSVDProjection

   n_freq = n_samples // 2 + 1
   num_channels = 2  # e.g. H1 and L1

   proj = FreqDomainSVDProjection(
       num_channels=num_channels,
       n_freq=n_freq,
       n_svd=n_svd,
       V=V,
   )

   # Forward pass: time-domain input -> SVD coefficients
   x = torch.randn(8, num_channels, n_samples)  # (batch, channels, time)
   coeffs = proj(x)
   print(f"Output shape: {coeffs.shape}")  # (8, 2 * 50) = (8, 100)

Per-channel vs shared projection
---------------------------------

By default, a single projection matrix is shared across all channels.
Set ``per_channel=True`` to use separate (independently trainable)
projections per channel, all initialized from the same V:

.. code-block:: python

   proj_shared = FreqDomainSVDProjection(
       num_channels=2, n_freq=n_freq, n_svd=50, V=V,
       per_channel=False,  # default: shared weights
   )

   proj_per_ch = FreqDomainSVDProjection(
       num_channels=2, n_freq=n_freq, n_svd=50, V=V,
       per_channel=True,  # separate weights per channel
   )

   # Both give the same initial output (weights start identical)
   x = torch.randn(4, 2, n_samples)
   y_shared = proj_shared(x)
   y_per_ch = proj_per_ch(x)
   assert torch.allclose(y_shared, y_per_ch, atol=1e-4)

Two-phase training with freeze/unfreeze
-----------------------------------------

A common training strategy is to first train the downstream network with
frozen SVD weights, then fine-tune the full model:

.. code-block:: python

   # Phase 1: freeze SVD, train downstream layers
   proj.freeze()
   for p in proj.parameters():
       assert not p.requires_grad

   # ... train downstream network ...

   # Phase 2: unfreeze SVD for end-to-end fine-tuning
   proj.unfreeze()
   for p in proj.parameters():
       assert p.requires_grad

   # ... fine-tune full model ...

Dense residual block for post-SVD processing
----------------------------------------------

The :class:`~ml4gw.nn.svd.dense.DenseResidualBlock` is designed for
processing SVD coefficients. It uses LayerNorm (not BatchNorm) to avoid
train/eval discrepancies in gravitational wave detection, where training
batches contain varying signal/noise ratios:

.. code-block:: python

   from ml4gw.nn.svd import DenseResidualBlock

   # Build a post-SVD processing network
   svd_dim = proj.output_dim  # num_channels * n_svd
   block = DenseResidualBlock(dim=svd_dim, dropout=0.1)

   coeffs = proj(x)
   processed = block(coeffs)
   print(f"Output shape: {processed.shape}")  # same as input: (8, 100)

Full pipeline example
---------------------

Putting it all together — from waveform generation to detection network:

.. code-block:: python

   import numpy as np
   import torch
   import torch.nn as nn
   from ml4gw.nn.svd import FreqDomainSVDProjection, DenseResidualBlock
   from ml4gw.waveforms import IMRPhenomD
   from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

   # 1. Generate waveform bank using ml4gw
   sample_rate = 2048.0
   duration = 4.0
   generator = TimeDomainCBCWaveformGenerator(
       approximant=IMRPhenomD(),
       sample_rate=sample_rate,
       duration=duration,
       f_min=20.0,
       f_ref=20.0,
       right_pad=0.5,
   )

   # Use more waveforms in practice (e.g. 5000+)
   n_waveforms = 200
   hc, hp = generator(
       chirp_mass=torch.distributions.Uniform(1.0, 2.0).sample((n_waveforms,)),
       mass_ratio=torch.distributions.Uniform(1.0, 4.0).sample((n_waveforms,)),
       chi1=torch.zeros(n_waveforms),
       chi2=torch.zeros(n_waveforms),
       distance=torch.full((n_waveforms,), 100.0),
       phic=torch.zeros(n_waveforms),
       inclination=torch.zeros(n_waveforms),
       mass_1=torch.distributions.Uniform(1.0, 3.0).sample((n_waveforms,)),
       mass_2=torch.distributions.Uniform(1.0, 3.0).sample((n_waveforms,)),
       s1z=torch.zeros(n_waveforms),
       s2z=torch.zeros(n_waveforms),
   )
   waveforms = hp.float().numpy()
   n_samples = waveforms.shape[-1]

   # 2. Compute SVD basis from waveform bank
   n_svd = 50
   V, s = FreqDomainSVDProjection.compute_basis(waveforms, n_svd=n_svd)

   # 3. Build projection layer
   n_freq = n_samples // 2 + 1
   num_ifos = 2
   proj = FreqDomainSVDProjection(
       num_channels=num_ifos, n_freq=n_freq, n_svd=n_svd, V=V,
   )

   # 4. Build downstream network
   feature_dim = proj.output_dim  # num_ifos * n_svd = 100
   net = nn.Sequential(
       DenseResidualBlock(feature_dim, dropout=0.1),
       DenseResidualBlock(feature_dim, dropout=0.1),
       nn.Linear(feature_dim, 1),
   )

   # 5. Forward pass
   x = torch.randn(16, num_ifos, n_samples)
   coeffs = proj(x)
   logits = net(coeffs)
   print(f"Detection logits: {logits.shape}")  # (16, 1)
