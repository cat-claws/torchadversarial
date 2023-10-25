# Torch adversarial
Optimizing inputs to models

## Installation
To install this small tool from the source code
```
pip install git+https://github.com/cat-claws/torchadversarial/
```

## How to use
```python
from torchadversarial import Fgsm

x = torch.rand(4, 3, 112, 112)
z = x.clone()
z.requires_grad = True
opt = Fgsm([z], epsilon = 0.1)

y = z.sum()
y.backward()

opt.step()
```
