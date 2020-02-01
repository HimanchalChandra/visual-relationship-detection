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
from util import AverageMeter, calculate_accuracy, Metric


def validate(model, loader, criterion, epoch, device, opt):

    model.eval()

    losses = AverageMeter()
    metric = AverageMeter()
    metric = Metric(opt.num_classes)
    with torch.no_grad():
        for i, (imgs, spatial_locations, word_vectors, targets_predicates, targets_confidences) in enumerate(loader):
            # compute outputs
            imgs, spatial_locations, word_vectors, targets_confidences, targets_predicates = imgs.to(device), spatial_locations.to(
                device), word_vectors.to(device),  targets_confidences.to(device), targets_predicates.to(device)
            confidences, predicates = model(imgs, spatial_locations, word_vectors)

            # compute loss
            loss = criterion(predicates, targets_predicates)

            metric.update(predicates, targets_predicates)
            losses.update(loss.item(), imgs.size(0))


    # show information
    recall = metric.compute_metrics()
    print('Validation set ({:d} samples): Average loss: {:.4f}\tRecall: {:.4f}'.format(losses.count, losses.avg, recall))
    return losses.avg, recall

