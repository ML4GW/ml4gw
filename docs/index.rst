.. ml4gw documentation master file, created by
   sphinx-quickstart on Mon Feb 19 09:02:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ml4gw's documentation!
=================================

.. note:: 
   This documentation is a work in progress!
   If you have any questions or suggestions, please feel free to create an `issue <https://github.com/ML4GW/ml4gw/issues>`_

ml4gw
=====
ml4gw is a library of `pytorch  <https://pytorch.org/docs/stable/index.html>`_ utilities 
for training neural networks in service of gravitational wave physics applications.

The code can be found on github at `<https://github.com/ML4GW/ml4gw>`_. Please see the 
usage examples and tutorial below for help on getting started.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   tutorials/ml4gw_tutorial

API:
----

.. autosummary::
   :toctree: api
   :caption: API
   :recursive:

   ml4gw
   ml4gw.utils

Projects
========
Currently, the following projects are using ml4gw to support their research:

* `Aframe <https://github.com/ml4gw/aframe>`_ - Gravitational wave detection of binary black hole mergers
* `AMPLFI <https://github.com/ml4gw/amplfi>`_ - Parameter estimation of gravitational wave signals


Development
===========
As this library is still very much a work in progress, 
we anticipate that novel use cases will encounter errors stemming from a lack of robustness. 
We encourage users who encounter these difficulties to file issues on GitHub, and we'll be happy to offer 
support to extend our coverage to new or improved functionality. 
We also strongly encourage ML users in the GW physics space to try their hand at working on these issues and joining on as collaborators! 
By bringing in new users with new use cases, we hope to develop this library into a truly general-purpose tool which makes 
DL more accessible for gravitational wave physicists everywhere.
   

Funding
=======
We are grateful for the support of the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for `Accelerating AI Algorithms for Data Driven Discovery (A3D3) <https://a3d3.ai">`_ under Cooperative Agreement No. `PHY-2117997 <https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997>`_.
