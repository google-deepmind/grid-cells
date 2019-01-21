# Vector-based navigation using grid-like representations in artificial agents

This package provides an implementation of the supervised learning experiments
in Vector-based navigation using grid-like representations in artificial agents,
as [published in Nature](https://www.nature.com/articles/s41586-018-0102-6)

Any publication that discloses findings arising from using this source code must
cite "Banino et al. "Vector-based navigation using grid-like representations in
artificial agents." Nature 557.7705 (2018): 429."

## Introduction

The grid-cell network is a recurrent deep neural network (LSTM). This network
learns to path integrate within a square arena, using simulated trajectories
modelled on those of foraging rodents. The network is required to update its
estimate of location and head direction using translational and angular velocity
signals which are provided as input. The output of the LSTM projects to place
and head direction units via a linear layer which is subject to regularization.
The vector of activities in the place and head direction units, corresponding to
the current position, was provided as a supervised training signal at each time
step.

The dataset needed to run this code can be downloaded from
[here](https://console.cloud.google.com/storage/browser/grid-cells-datasets).

The files contained in the repository are the following:

*   `train.py` is where the training and logging loop happen; The file comes
    with the flags defined in Table 1 of the paper. In order to run this file
    you will need to specify where the dataset is stored and where you want to
    save the results. The results are saved in PDF format and they contains the
    ratemaps and the spatial autocorrelagram order by grid score. The units are
    ordered from higher to lower grid score. Only the last evaluation is saved.
    Please note that given random seeds results can vary between runs.

*   `data_reader.py` read the TFRecord and returns a ready to use batch, which
    is already shuffled.

*   `model.py` contains the grid-cells network

*   `scores.py` contains all the function for calculating the grid scores and
    doing the plotting.

*   `ensembles.py` contains the classes to generate the targets for training of
    the grid-cell networks.

## Train

The implementation requires an installation of
[TensorFlow](https://www.tensorflow.org/) version 1.12, and
[Sonnet](https://github.com/deepmind/sonnet) version 1.27.

```shell
$ virtualenv env
$ source env/bin/activate
$ pip install --upgrade numpy==1.13.3
$ pip install --upgrade tensorflow==1.12.0-rc0
$ pip install --upgrade dm-sonnet==1.27
$ pip install --upgrade scipy==1.0.0
$ pip install --upgrade matplotlib==1.5.2
$ pip install --upgrade tensorflow-probability==0.5.0
$ pip install --upgrade wrapt==1.9.0
```

An example training script can be executed from a python interpreter:

```shell
$ python train.py --task_root='path/to/datasets/root/folder' --saver_results_directory='path/to/results/folder'
```

Disclaimer: This is not an official Google product.

