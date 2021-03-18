# TorchRC
[![PyPI version fury.io](https://badge.fury.io/py/torch-rc.svg)](https://pypi.python.org/pypi/torch-rc/)

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

 * Leaky Echo State Network (`torch_rc.nn.LeakyESN`)
 * Leaky Echo State Network with Ring or Multiring Reservoir (`torch_rc.nn.MultiringESN`)

More models are coming.

### Optimizers

TorchRC allows to train the reservoir models either in closed form or with the standard PyTorch optimizers.
Exact incremental closed form techniques are supported in order to support those scenarios in which it is not feasible to hold all the network states in memory.
Training on the GPU is also supported.

## Installation

    pip3 install torch-rc

## Example

You can find example scripts in the [examples/](examples/) folder. 
