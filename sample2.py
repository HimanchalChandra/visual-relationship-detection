import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np

import json
import torch.nn as nn
import torch
import argparse
import random
import cv2
from models import MFURLN

# model = vgg16(pretrained=True)
# modules = list(model.children())[:-2]  
# base_net = nn.Sequential(*modules)


# define model
model = MFURLN(num_classes=70)
model = model.to('cuda')
model = nn.DataParallel(model)

# load pretrained weights
checkpoint = torch.load('./snapshots/model26.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model Restored")

