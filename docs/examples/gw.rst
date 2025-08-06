Gravitational-wave projection and SNR calculation
=================================================

.. code-block:: python

   """
   Project a batch of waveform polarizations onto the Hanford,
   Livingston, and Virgo interferometers to compute the observed
   gravitational wave strain.
   """

   from ml4gw.gw import get_ifo_geometry, compute_observed_strain
   from ml4gw.distributions import Cosine
   from torch.distributions import Uniform

   dec = Cosine()
   psi = Uniform(0, torch.pi)
   phi = Uniform(-torch.pi, torch.pi)

   # Get the interferometer geometry
   ifos = ["H1", "L1", "V1"]
   tensors, vertices = get_ifo_geometry(*ifos)

   # The following assumes that the plus and cross polarizations
   # of the gravitational wave have already been computed by
   # some method; e.g., using the `TimeDomainCBCWaveformGenerator`
   # from the `ml4gw.waveforms` module. `sample_rate` is the sample
   # rate at which the polarizations were generated.
   waveforms = compute_observed_strain(
      dec=dec.sample((num_waveforms,)),
      psi=psi.sample((num_waveforms,)),
      phi=phi.sample((num_waveforms,)),
      detector_tensors=tensors,
      detector_vertices=vertices,
      sample_rate=sample_rate,
      cross=hc,
      plus=hp,
   )

.. code-block:: python

   """
   Compute the signal-to-noise ratio (SNR) of a batch of waveforms
   relative to a given noise power spectral density (PSD).
   """

   from ml4gw.gw import compute_network_snr

   # Assume `waveforms` is a batch of waveforms with shape
   # (num_waveforms, num_detectors, num_samples) and 
   # `psd` is a batch of power spectral densities. See the
   # docstring for details on the allowed PSD shapes.

   # Highpass determines the minimum frequency for the SNR calculation.
   highpass = 20

   # Sample rate of the waveforms
   sample_rate = 2048

   snr = compute_network_snr(
      responses=waveforms,
      psd=psd,
      sample_rate=sample_rate,
      highpass=highpass,
   )
