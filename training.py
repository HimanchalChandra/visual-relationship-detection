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
from util import AverageMeter, calculate_accuracy


def train(model, loader, criterion, optimizer, epoch, device, log_interval):

    model.train()
    
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (imgs, spatial_locations, word_vectors, targets) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(
            device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
        outputs = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), imgs.size(0))
        accuracies.update(acc, imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show information
        if (i+1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, losses.count, len(loader.dataset), 100. * (i + 1) / len(loader), avg_loss))
            train_loss = 0.0

    # show information
    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        losses.count, losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  