---
title: 'ml4gw: PyTorch utilities for training neural networks in gravitational wave physics applications'
tags:
  - Python
  - PyTorch
  - machine learning
  - gravitational waves
  - signal processing
authors:
  - name: William Benoit
    orcid: 0000-0003-4750-9413
    affiliation: 1
    corresponding: true
  - name: Ethan Marx
    orcid: 0009-0000-4183-7876
    affiliation: "2, 3"
  - name: Deep Chatterjee
    orcid: 0000-0003-0038-5468
    affiliation: 3
  - name: Ravi Kumar
    affiliation: 4
affiliations:
 - name: University of Minnesota, USA
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
 - name: MIT LIGO Laboratory, USA
   index: 3
 - name: Indian Institute of Technology Bombay, India
   index: 4
date: 1 April 2025
bibliography: references.bib
---

# Summary

`ml4gw` is a lightweight, PyTorch-based library designed to support the development, scaling, and deployment of machine learning (ML) models for gravitational-wave (GW) data analysis. 
Built to provide familiar functionality to anyone used to working with standard CPU-based GW libraries, `ml4gw` lowers the barrier to entry for GW researchers who want to incorporate ML into their work and take advantage of optimized hardware. 
Its design focuses on bridging the gap between domain-specific GW signal processing and general-purpose ML development, allowing researchers to build the tools needed to accomplish their scientific goals without getting stuck on infrastructure overhead. 
By simplifying the interface between domain science and model training, `ml4gw` accelerates the creation of robust, physically-informed ML pipelines for gravitational-wave astrophysics.

# Statement of need

Machine-learning algorithms are well-suited for applications in GW astrophysics due to the field's abundance of high-quality data.
The existence of multiple independent GW detectors allows for effectively unlimited combinations of noise samples via time-shifts, and, for the most common signal morphologies, the high-fidelity simulations provided by General Relativity allow as many signal samples as desired.
However, the standard GW libraries are not designed for the scale or speed desired for ML development; conversely, standard ML libraries lack the domain-specific preprocessing functions and waveform handling tools required to train models for GW applications.

`ml4gw` addresses this gap by re-writing common GW processing steps as PyTorch [@pytorch] modules, taking advantage of the natural parallelization that comes with PyTorch's batch processing, and adding the option to accelerate these steps using GPUs and other coprocessors supported by the PyTorch framework.
From libraries such as GWpy [@gwpy], `ml4gw` re-implements power spectral density estimation, signal-to-noise ratio calculation, whitening filters, and Q-transforms.
Like bilby [@bilby], `ml4gw` provides the functionality to sample from astrophysical parameter distributions, which can then be used to simulate waveforms, mimicking the simulation features of lalsuite [@lalsuite].
`ml4gw` has available basic compact binary merger waveform families used in online searches and inference (TaylorF2, IMRPhenomD, and IMRPhenomPv2), as well as sine-gaussian waveforms for capturing unmodeled GW signals, with more complex waveforms planned for the future.
All of these modules have been designed to work with batches of multi-channel time-series data, run on accelerated hardware, and be composable so that the output of one function can easily become the input of another.

Additionally, `ml4gw` contains a number of general utility features.

- Efficient out-of-memory dataloading to scale the quantity of data used for training
- Random sampling of windows from batches of multi-channel time-series data
- On-the-fly signal generation for efficient scaling of training data
- Cubic spline interpolation to supplement PyTorch's existing interpolation functions
- Basic out-of-the-box neural network architectures to streamline the startup process for new users
- Stateful modules for handling streaming data

All implementations are fully differentiable, allowing algorithms to employ physically-motivated loss functions.
These features make it possible to train models on large quantities realistic detector data, perform data augmentation consistent with physical priors, and evaluate results in a way that is directly comparable with existing pipelines.

`ml4gw` is used to support the development of multiple GW analyses. 
It has been integrated into `DeepClean` [@deepclean], a noise-subtraction pipeline; `Aframe` [@aframe], a search pipeline for gravitational waves from compact binary mergers; GWAK [@gwak], a gravitational-wave anomaly detection pipeline; and AMPLFI [@amplfi], a gravitational-wave parameter estimation pipelines.
`ml4gw` has enabled these algorithms to efficiently train ML models at scale and deploy models on real-time streaming data, while the use of a standardized tool set has allowed for easier communication between developers.
The library is actively developed and maintained, with thorough unit testing, and the authors welcome contributions and collaborations.

# Acknowledgements

This work was supported by the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for Accelerating AI Algorithms for Data Driven Discovery (A3D3) under Cooperative Agreement No. PHY-2117997.

# References
