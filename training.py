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


def train(model, loader, criterion, optimizer, epoch, device, log_interval):
    model.train()
    train_loss = 0.0
    N_count = 0
    correct = 0
    losses = []
    for i, (imgs, spatial_locations, word_vectors, targets) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(
            device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
        N_count += imgs.size(0)
        outputs = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        train_loss += loss.item()

        # to compute accuracy
        outputs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(targets.view_as(preds)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show information
        if (i+1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, N_count, len(loader.dataset), 100. * (i + 1) / len(loader), avg_loss))
            train_loss = 0.0

    # show information
    acc = 100. * (correct / N_count)
    average_loss = sum(losses)/len(loader)
    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        N_count, average_loss, acc))
    return average_loss, acc