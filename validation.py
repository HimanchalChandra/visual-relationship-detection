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
import torch.nn.functional as F
import json
import torch.nn as nn
import argparse
import random
import cv2
from util import AverageMeter, calculate_accuracy, Metric


def validate(model, loader, criterion, epoch, device, opt):

    model.eval()

    losses = AverageMeter()
    metric = Metric(opt.num_classes)
    with torch.no_grad():
         for i, (imgs, spatial_locations, word_vectors, targets_predicates, targets_confidences, rois_sub, rois_obj) in enumerate(loader):
            # compute outputs
            imgs, spatial_locations, word_vectors, targets_confidences, targets_predicates, rois_sub, rois_obj = imgs.to(device), spatial_locations.to(
            device), word_vectors.to(device),  targets_confidences.to(device), targets_predicates.to(device), rois_sub.to(device), rois_obj.to(device)
            predicates = model(imgs, spatial_locations, word_vectors, rois_sub, rois_obj)


            # compute loss
            predicates = F.softmax(predicates, dim=1)
            loss = criterion[1](predicates, targets_predicates)

            metric.update(predicates, targets_predicates)
            losses.update(loss.item(), imgs.size(0))


    # show information
    recall, precision = metric.compute_metrics()
    print('Validation set ({:d} samples): Average loss: {:.4f}\tRecall: {:.4f}\tPrecision: {:.4f}'.format(losses.count, losses.avg, recall, precision))
    return losses.avg, recall

