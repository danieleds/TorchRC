# TorchRC
[![PyPI version fury.io](https://badge.fury.io/py/torch-rc.svg)](https://pypi.python.org/pypi/torch-rc/) [![Python package](https://github.com/danieleds/TorchRC/actions/workflows/python-package.yml/badge.svg)](https://github.com/danieleds/TorchRC/actions/workflows/python-package.yml)

An organized collection of Reservoir Computing models and techniques that is well-integrated within the PyTorch API.

> **WARNING**: Work in progress!

* [What's inside](#whats-inside)
  * [Models](#models)
  * [Optimizers](#optimizers)
* [Installation](#installation)
* [Example](#example)

## What's inside

### Models

At the moment, the library contains an implementation of:

 * **(Leaky/Deep) Echo State Network** (`torch_rc.nn.LeakyESN`)
 * **(Leaky/Deep) Echo State Network with Ring or Multiring Reservoir** (`torch_rc.nn.MultiringESN`)

More models are coming.

### Optimizers

TorchRC allows to train the reservoir models either in closed form or with the standard PyTorch optimizers.
Exact incremental closed form techniques are supported in order to support those scenarios in which it is not feasible to hold all the network states in memory.
Training on the GPU is also supported.

Currently supported optimizers:

 * **Ridge Classification** (`torch_rc.optim.RidgeClassification`): for trainin a readout in closed-form for classification problems. 
 * **Ridge Regression** (`torch_rc.optim.RidgeRegression`): for trainin a readout in closed-form for regression problems. 
 * **Ridge Incremental Classification** (`torch_rc.optim.RidgeIncrementalClassification`): for training a readout in closed-form for classification problems,
   passing data in multiple separate calls (e.g., for when your collection of states do not fit in memory).
 * **Ridge Incremental Regression** (`torch_rc.optim.RidgeIncrementalRegression`): for training a readout in closed-form for regression problems,
   passing data in multiple separate calls (e.g., for when your collection of states do not fit in memory).

## Installation

    pip3 install torch-rc

## Example

You can find example scripts in the [examples/](examples/) folder. 
