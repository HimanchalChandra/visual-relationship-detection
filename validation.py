import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from PIL import Image, ImageFont, ImageDraw
import json
import torch.nn as nn
import argparse
import random
import cv2


def validate(model, loader, criterion, device):
    model.eval()
    N_count = 0
    correct = 0
    losses = []
    for imgs, spatial_locations, word_vectors, targets in loader:
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
        N_count += imgs.size(0)
        outputs = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        # to compute accuracy
        outputs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(targets.view_as(preds)).sum().item()

    # show information
    acc = 100. * (correct / N_count)
    average_loss = sum(losses)/len(loader)
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(N_count, average_loss, acc))
    return average_loss, acc
