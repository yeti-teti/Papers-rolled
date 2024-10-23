#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imoprts
import torch
from torch import nn, Size


# ### Layer Normalization

# In[10]:


class LayerNorm(nn.Module):

    def __init__(
            self, 
            normalized_shape: int | list[int] | Size,
            eps: float = 1e-5, 
            elementwise_affine: bool = True
    ):
        super(LayerNorm, self).__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):

        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        dims = [ -(i+1) for i in range(len(self.normalized_shape)) ]

        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)

        var = mean_x2 - mean ** 2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias
        
        return x_norm


# In[ ]:





# In[ ]:





# ### Dropout

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




