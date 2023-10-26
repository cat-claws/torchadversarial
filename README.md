# Torch adversarial
Optimizing inputs to models

## Installation
To install this small tool from the source code
```
pip install git+https://github.com/cat-claws/torchadversarial/
```

## Advantages
You can use these attack codes for more tasks than classification. For example, FGSM can be used to attack object detection etc.

## How to use
```python
import torch
from torchadversarial import Fgsm

x = torch.rand(4, 3, 112, 112)
z = x.clone()
z.requires_grad = True
opt = Fgsm([z], epsilon = 0.1)

y = z.sum()
y.backward()

opt.step()
```

To use as an adversarial attack
```python
import torch
from torchadversarial import Fgsm, Attack

x = torch.rand(4, 3, 112, 112)

for x_ in Attack(Fgsm, [x], epsilon = 0.1, foreach = False, maximize = True):
    y = torch.sum(x_[0])
    y.backward()

# print(x_[0])
```
Note that ```x_[0]``` will be your adversarial example. Like other optimizers in ```torch.optim```, the input parameters, _e.g._,  ```[x]``` must be an iterable containing tensors. Thus, we must take ```x_[0]``` as our output in this example above.

