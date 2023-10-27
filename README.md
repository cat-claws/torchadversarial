# Torch Adversarial - Optimizing Inputs to Models

[![GitHub license](https://img.shields.io/github/license/cat-claws/torchadversarial.svg)](https://github.com/cat-claws/torchadversarial/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/torchadversarial.svg)](https://pypi.org/project/torchadversarial/)

Torch Adversarial is a versatile toolkit for optimizing inputs to models. It provides a wide range of attack methods that can be used for various tasks beyond classification, including object detection and more.

## Installation
To install Torch Adversarial from PyPI, simply run the following command:

```bash
pip install torchadversarial
```

To install Torch Adversarial from the source code, use the following command:

```bash
pip install git+https://github.com/cat-claws/torchadversarial/
```

## Advantages
Torch Adversarial offers attack codes that are not limited to classification tasks. For example, FGSM (Fast Gradient Sign Method) can be employed to attack object detection models and more.

## Table of Contents
- [Installation](#installation)
- [Advantages](#advantages)
- [How to Use](#how-to-use)
  - [Basic Usage](#basic-usage)
  - [Using as an Adversarial Attack](#using-as-an-adversarial-attack)


## How to use
### Basic Usage
You can use Torch Adversarial to optimize input tensors as follows:

```python
import torch
from torchadversarial import Fgsm

# Create an input tensor
x = torch.rand(4, 3, 112, 112)

# Clone the input tensor and enable gradient computation
z = x.clone()
z.requires_grad = True

# Initialize the FGSM optimizer with epsilon (perturbation magnitude)
opt = Fgsm([z], epsilon=0.1)

# Define a simple objective function (for example, sum of tensor elements)
y = z.sum()

# Backpropagate through the optimizer
y.backward()

# Perform a step to generate the adversarial example
opt.step()
```
### Using as an Adversarial Attack
You can utilize Torch Adversarial as an adversarial attack as shown below:

```python
import torch
from torchadversarial import Fgsm, Attack

# Create an input tensor
x = torch.rand(4, 3, 112, 112)

# Apply the FGSM attack to the input tensor
for x_ in Attack(Fgsm, [x], epsilon=0.1, foreach=False, maximize=True):
    # Define an objective function (e.g., sum of tensor elements)
    y = torch.sum(x_[0])
    y.backward()

# x_[0] now contains your adversarial example
# print(x_[0])
```

Please note that in the example above, ```x_[0]``` represents your adversarial example. Similar to other optimizers in ```torch.optim```, the input parameters (e.g., ```[x]```) must be an iterable containing tensors, and thus, we extract the adversarial example as ```x_[0]```.

With Torch Adversarial, you can efficiently optimize inputs to your models and perform adversarial attacks for a wide range of applications.
