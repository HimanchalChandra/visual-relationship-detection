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
<<<<<<< HEAD
    N_count = 0
    correct = 0
    losses = []
=======
    losses = AverageMeter()
    metric = Metric(opt.num_classes)
>>>>>>> v5
    for i, (imgs, spatial_locations, word_vectors, targets_confidences, targets_predicates) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(
            device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
<<<<<<< HEAD
        N_count += imgs.size(0)
=======
>>>>>>> v5
        confidences, predicates = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss1 = criterion(confidences, targets_confidences)
        loss2 = criterion(predicates, targets_predicates)
<<<<<<< HEAD
        total_loss = loss1 + loss2
        losses.append(total_loss.item())
        train_loss += total_loss.item()
=======
        tot_loss = loss1 + loss2
        train_loss += tot_loss.item()
>>>>>>> v5

        losses.update(train_loss.item(), imgs.size(0))
        metric.update(predicates, targets_predicates)

        optimizer.zero_grad()
        total_loss.backward()
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
<<<<<<< HEAD
        N_count, average_loss, acc))
    return average_loss, acc
=======
        losses.count, losses.avg, recall * 100))

    return losses.avg, recall 
>>>>>>> v5
