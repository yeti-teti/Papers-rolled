import math

import numpy as np

import torch
from torch import nn

from tinygrad import Tensor
from tinygrad import nn as nn_tg


## Custom
input = torch.tensor([-3., -2., -1., 0., 1., 2., 3.])
gelu_py = 0.5 * input * (1.0 + torch.tanh( math.sqrt(2.0/math.pi) * (input + (0.044715 * (np.power(input, 3.0) )) )) ) 
print(gelu_py)

## Pytorch
#input = torch.randn(2)
input = torch.tensor([[-3., -2., -1., 0., 1., 2., 3.]])
print(f"Input:{input}")
gelu_pt = nn.GELU()
print(gelu_pt(input))
gelu_pt = nn.GELU(approximate='tanh')
print(gelu_pt(input))


## Tinygrad
print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy()) # GELU
print(Tensor([[-3., -2., -1., 0., 1., 2., 3.]]).quick_gelu().numpy()) # Quick_GELU
