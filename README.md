# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/ML4GW/ml4gw/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                      |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------ | -------: | -------: | ------: | --------: |
| ml4gw/\_\_init\_\_.py                     |        1 |        0 |    100% |           |
| ml4gw/augmentations.py                    |       19 |        0 |    100% |           |
| ml4gw/constants.py                        |       12 |        0 |    100% |           |
| ml4gw/dataloading/\_\_init\_\_.py         |        3 |        0 |    100% |           |
| ml4gw/dataloading/chunked\_dataset.py     |       45 |        2 |     96% |    61, 96 |
| ml4gw/dataloading/hdf5\_dataset.py        |       73 |        4 |     95% |86, 197-201 |
| ml4gw/dataloading/in\_memory\_dataset.py  |       63 |        4 |     94% |157-158, 201-202 |
| ml4gw/distributions.py                    |      146 |        2 |     99% |   114-115 |
| ml4gw/gw.py                               |       99 |        0 |    100% |           |
| ml4gw/nn/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| ml4gw/nn/autoencoder/\_\_init\_\_.py      |        3 |        3 |      0% |       1-3 |
| ml4gw/nn/autoencoder/base.py              |       44 |       44 |      0% |      1-94 |
| ml4gw/nn/autoencoder/convolutional.py     |       55 |       55 |      0% |     1-159 |
| ml4gw/nn/autoencoder/skip\_connection.py  |       30 |       30 |      0% |      1-47 |
| ml4gw/nn/autoencoder/utils.py             |       12 |       12 |      0% |      1-15 |
| ml4gw/nn/norm.py                          |       51 |        2 |     96% |   87, 106 |
| ml4gw/nn/resnet/\_\_init\_\_.py           |        2 |        0 |    100% |           |
| ml4gw/nn/resnet/resnet\_1d.py             |      142 |        1 |     99% |       331 |
| ml4gw/nn/resnet/resnet\_2d.py             |      142 |        1 |     99% |       331 |
| ml4gw/nn/streaming/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| ml4gw/nn/streaming/online\_average.py     |       49 |        4 |     92% |82, 90-91, 115 |
| ml4gw/nn/streaming/snapshotter.py         |       35 |        2 |     94% |   96, 110 |
| ml4gw/spectral.py                         |      147 |        0 |    100% |           |
| ml4gw/transforms/\_\_init\_\_.py          |       10 |        0 |    100% |           |
| ml4gw/transforms/iirfilter.py             |       18 |        0 |    100% |           |
| ml4gw/transforms/pearson.py               |       28 |        3 |     89% |47, 52, 59 |
| ml4gw/transforms/qtransform.py            |      172 |       14 |     92% |149, 328, 336-338, 345-354 |
| ml4gw/transforms/scaler.py                |       32 |        0 |    100% |           |
| ml4gw/transforms/snr\_rescaler.py         |       40 |       16 |     60% |51-67, 74-88 |
| ml4gw/transforms/spectral.py              |       32 |        0 |    100% |           |
| ml4gw/transforms/spectrogram.py           |       58 |        0 |    100% |           |
| ml4gw/transforms/spline\_interpolation.py |      102 |        0 |    100% |           |
| ml4gw/transforms/transform.py             |       38 |        0 |    100% |           |
| ml4gw/transforms/waveforms.py             |       42 |        0 |    100% |           |
| ml4gw/transforms/whitening.py             |       48 |        0 |    100% |           |
| ml4gw/types.py                            |       18 |        0 |    100% |           |
| ml4gw/utils/interferometer.py             |       20 |        0 |    100% |           |
| ml4gw/utils/slicing.py                    |      103 |        0 |    100% |           |
| ml4gw/waveforms/\_\_init\_\_.py           |        2 |        0 |    100% |           |
| ml4gw/waveforms/adhoc/\_\_init\_\_.py     |        2 |        0 |    100% |           |
| ml4gw/waveforms/adhoc/ringdown.py         |       36 |       29 |     19% |18-25, 57-109 |
| ml4gw/waveforms/adhoc/sine\_gaussian.py   |       37 |        0 |    100% |           |
| ml4gw/waveforms/cbc/\_\_init\_\_.py       |        3 |        0 |    100% |           |
| ml4gw/waveforms/cbc/coefficients.py       |       13 |        0 |    100% |           |
| ml4gw/waveforms/cbc/phenom\_d.py          |      326 |        1 |     99% |       628 |
| ml4gw/waveforms/cbc/phenom\_d\_data.py    |        4 |        0 |    100% |           |
| ml4gw/waveforms/cbc/phenom\_p.py          |      284 |        0 |    100% |           |
| ml4gw/waveforms/cbc/taylorf2.py           |      105 |        0 |    100% |           |
| ml4gw/waveforms/cbc/utils.py              |       48 |        0 |    100% |           |
| ml4gw/waveforms/conversion.py             |       81 |        0 |    100% |           |
| ml4gw/waveforms/generator.py              |      103 |        0 |    100% |           |
|                                 **TOTAL** | **2980** |  **229** | **92%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/ML4GW/ml4gw/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/ML4GW/ml4gw/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ML4GW/ml4gw/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/ML4GW/ml4gw/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FML4GW%2Fml4gw%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/ML4GW/ml4gw/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.