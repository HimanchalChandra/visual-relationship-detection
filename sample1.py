import torch
import torch.nn as nn
from torchvision import models
import json
import numpy as np
import torch.nn.functional as F
from opts import parse_opts
from torch.nn import CrossEntropyLoss
import os


class VisionModule(nn.Module):
	""" Vision Moddule"""
	def __init__(self):
		super(VisionModule, self).__init__()
		vgg = models.vgg16(pretrained=False)
		modules = list(vgg.children())[:-1]
		self.resnet_backbone = nn.Sequential(*modules)
		self.fc = nn.Linear(512, 5)

	def forward(self, x):
		x = self.resnet_backbone(x)
		print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = F.relu(x)
		return x


model = VisionModule()
print(model)


x = torch.Tensor(2,3,224,224)

print(x.shape)
out = model(x)
print(out.shape)

# criterion = CrossEntropyLoss()

# targets = [0,1]
# targets = torch.Tensor(targets)

# print(out.type())
# targets = targets.type(torch.LongTensor)

# print(targets)

# loss1 = criterion(out, targets)
# print(loss1)
