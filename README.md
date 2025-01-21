# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/ML4GW/ml4gw/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                      |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------ | -------: | -------: | ------: | --------: |
| ml4gw/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| ml4gw/augmentations.py                    |       19 |        0 |    100% |           |
| ml4gw/constants.py                        |       12 |        0 |    100% |           |
| ml4gw/dataloading/\_\_init\_\_.py         |        3 |        0 |    100% |           |
| ml4gw/dataloading/chunked\_dataset.py     |       45 |        2 |     96% |    61, 96 |
| ml4gw/dataloading/hdf5\_dataset.py        |       66 |        3 |     95% |   169-173 |
| ml4gw/dataloading/in\_memory\_dataset.py  |       63 |        4 |     94% |157-158, 201-202 |
| ml4gw/distributions.py                    |       61 |       10 |     84% |41-43, 58-64, 106-107, 136 |
| ml4gw/gw.py                               |       90 |        9 |     90% |48, 128-129, 351-361 |
| ml4gw/nn/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| ml4gw/nn/autoencoder/\_\_init\_\_.py      |        3 |        3 |      0% |       1-3 |
| ml4gw/nn/autoencoder/base.py              |       44 |       44 |      0% |      1-94 |
| ml4gw/nn/autoencoder/convolutional.py     |       55 |       55 |      0% |     1-159 |
| ml4gw/nn/autoencoder/skip\_connection.py  |       30 |       30 |      0% |      1-47 |
| ml4gw/nn/autoencoder/utils.py             |       12 |       12 |      0% |      1-15 |
| ml4gw/nn/norm.py                          |       51 |        2 |     96% |   87, 106 |
| ml4gw/nn/resnet/\_\_init\_\_.py           |        2 |        0 |    100% |           |
| ml4gw/nn/resnet/resnet\_1d.py             |      142 |        8 |     94% |72, 319-320, 328-332 |
| ml4gw/nn/resnet/resnet\_2d.py             |      142 |        6 |     96% |69, 328-332 |
| ml4gw/nn/streaming/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| ml4gw/nn/streaming/online\_average.py     |       49 |        4 |     92% |82, 90-91, 115 |
| ml4gw/nn/streaming/snapshotter.py         |       35 |        2 |     94% |   96, 110 |
| ml4gw/spectral.py                         |      147 |        2 |     99% |  324, 524 |
| ml4gw/transforms/\_\_init\_\_.py          |        9 |        0 |    100% |           |
| ml4gw/transforms/pearson.py               |       28 |        3 |     89% |48, 53, 60 |
| ml4gw/transforms/qtransform.py            |      172 |       14 |     92% |149, 328, 336-338, 345-354 |
| ml4gw/transforms/scaler.py                |       32 |        0 |    100% |           |
| ml4gw/transforms/snr\_rescaler.py         |       37 |       16 |     57% |42-58, 65-75 |
| ml4gw/transforms/spectral.py              |       32 |        1 |     97% |        95 |
| ml4gw/transforms/spectrogram.py           |       57 |        0 |    100% |           |
| ml4gw/transforms/spline\_interpolation.py |      105 |        4 |     96% |261-263, 285, 364 |
| ml4gw/transforms/transform.py             |       38 |        0 |    100% |           |
| ml4gw/transforms/waveforms.py             |       42 |        2 |     95% |     58-59 |
| ml4gw/transforms/whitening.py             |       48 |        1 |     98% |       258 |
| ml4gw/types.py                            |       18 |        0 |    100% |           |
| ml4gw/utils/interferometer.py             |       20 |        1 |     95% |        45 |
| ml4gw/utils/slicing.py                    |      103 |        5 |     95% |62-64, 178, 296 |
| ml4gw/waveforms/\_\_init\_\_.py           |        2 |        0 |    100% |           |
| ml4gw/waveforms/adhoc/\_\_init\_\_.py     |        2 |        0 |    100% |           |
| ml4gw/waveforms/adhoc/ringdown.py         |       36 |       29 |     19% |18-25, 57-109 |
| ml4gw/waveforms/adhoc/sine\_gaussian.py   |       37 |        0 |    100% |           |
| ml4gw/waveforms/cbc/\_\_init\_\_.py       |        3 |        0 |    100% |           |
| ml4gw/waveforms/cbc/phenom\_d.py          |      346 |        7 |     98% |66, 178, 229, 278, 440, 517, 672 |
| ml4gw/waveforms/cbc/phenom\_d\_data.py    |        4 |        0 |    100% |           |
| ml4gw/waveforms/cbc/phenom\_p.py          |      284 |        2 |     99% |   84, 618 |
| ml4gw/waveforms/cbc/taylorf2.py           |      105 |        1 |     99% |        63 |
| ml4gw/waveforms/conversion.py             |       81 |        1 |     99% |        87 |
| ml4gw/waveforms/generator.py              |       18 |        0 |    100% |           |
|                                 **TOTAL** | **2732** |  **283** | **90%** |           |


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