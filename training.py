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
from util import calculate_accuracy
from utils import AverageMeter, calculate_accuracy



def train(model, loader, criterion, optimizer, epoch, device, epoch_logger, batch_logger):

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (imgs, spatial_locations, word_vectors, targets) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets = imgs.to(
            device), spatial_locations.to(device), word_vectors.to(device),  targets.to(device)
        outputs = model(imgs, spatial_locations, word_vectors)

        # compute loss
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), imgs.size(0))
        accuracies.update(acc, imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(loader),
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })


    # # show information
    # acc = 100. * (correct / N_count)
    # average_loss = sum(losses)/len(loader)
    # print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
    #     N_count, average_loss, acc))
    # return average_loss, acc