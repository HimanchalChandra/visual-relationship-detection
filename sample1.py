import torch
from torchvision import transforms, utils
import numpy as np


class Scale(object):
    """Feature Scaling"""

    def __call__(self, sample):
        sample = sample/224
        return sample


scale = Scale()
        
a = torch.Tensor([224,0])
print(scale(a))


a = [1,2,3]
a = [i/ 3 for i in a]
print(a)