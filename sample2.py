import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torchvision.models import vgg16
from PIL import Image, ImageFont, ImageDraw
from dataset import VrdDataset
import json
import torch.nn as nn

import argparse
import random
import cv2
from models import MFURLN
from shapely.geometry import box
from shapely.ops import cascaded_union
from retinanet.obj_det_retinanet import ObjDetRetinanet
from utils.util import calc_iou,calc_intersection
from opts import parse_opts
# model = vgg16(pretrained=True)
# modules = list(model.children())[:-2]  
# base_net = nn.Sequential(*modules)


# define model
model = MFURLN(num_classes=70)
model = model.to('cuda')


# load pretrained weights
checkpoint = torch.load('./snapshots/model26.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print("Model Restored")

