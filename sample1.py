import torch
import torch.nn as nn
from torchvision import models
import json
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class VisionModule(nn.Module):
    """ Vision Moddule"""

    def __init__(self):
        super(VisionModule, self).__init__()
        resnet = models.vgg16(pretrained=False)
        # classifier = nn.Sequential(OrderedDict([
        #                       ('dropout',nn.Dropout(0.5)),
        #                       ('fc1', nn.Linear(10, 50))
        #                       ]))

        # resnet.classifier = classifier

        print(resnet)
    
        modules = list(resnet.children())[:-1]
        self.resnet_backbone = nn.Sequential(*modules)
       

        self.fc1 = nn.Linear(512, 4096)
        self.fc1_final = nn.Linear(4096, 5)

    def forward(self, x):
        x = self.resnet_backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_final(x)
        return x









model = VisionModule()

x = torch.Tensor(2,3,64,64)
# x = model(x)
# print(x.shape)

