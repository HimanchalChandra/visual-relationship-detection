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
from torchvision.models import resnet50
import argparse
import random
import cv2
from model import Net

def main():
    
    checkpoint = torch.load('./snapshots/model_acc_83.pth')
    model = checkpoint['model_state_dict']
    for k, v in model.items():
        print(k)

if __name__ == "__main__":
    main()
