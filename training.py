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
from util import AverageMeter, Metric


def train(model, loader, criterion, optimizer, epoch, device, log_interval):

    model.train()
    
    train_loss = 0.0
    losses = AverageMeter()
    metric = Metric(opt.num_classes)
    for i, (imgs, spatial_locations, word_vectors, targets_confidences, targets_predicates) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(
            device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
        confidences, predicates = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss1 = criterion(confidences, targets_confidences)
        loss2 = criterion(predicates, targets_predicates)
        tot_loss = loss1 + loss2
        train_loss += tot_loss.item()

        losses.update(train_loss.item(), imgs.size(0))
        metric.update(predicates, targets_predicates)

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
    recall = metric.compute_metrics()
    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        losses.count, losses.avg, recall * 100))

    return losses.avg, recall 