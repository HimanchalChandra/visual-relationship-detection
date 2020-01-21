import torch
import torch.nn as nn
from torchvision import models
import json
import numpy as np
import torch.nn.functional as F
from opts import parse_opts
import torchvision.ops as ops
import os

self.roi_pool = ops.RoIPool(output_size=(7, 7), spatial_scale=0.03125)