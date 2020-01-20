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


# def loss_weights(opt):
#     weights_pred = [1.5 for _ in range(opt.num_classes - 1)]
#     weights_pred.append(0.01)
#     weights_pred = torch.tensor(weights_pred).cuda()

#     weights_conf = [0.01, 1.5]
#     weights_conf = torch.tensor(weights_conf).cuda()

#     return weights_conf, weights_pred


def train(model, loader, criterion, optimizer, epoch, device, opt):

    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    metric = Metric(opt.num_classes)

    #weights_conf, weights_pred = loss_weights(opt)
    for i, (imgs, spatial_locations, word_vectors, targets_predicates, targets_confidences) in enumerate(loader):
        # compute outputs
        imgs, spatial_locations, word_vectors, targets_confidences, targets_predicates = imgs.to(device), spatial_locations.to(
            device), word_vectors.to(device),  targets_confidences.to(device), targets_predicates.to(device)
        confidences, predicates = model(imgs, spatial_locations, word_vectors)

        print(predicates)

        # compute loss
        loss1 = criterion[0](confidences, targets_confidences)
        # loss1 = (loss1 * weights_conf).mean()

        loss2 = criterion[1](predicates, targets_predicates)
        # loss2 = (loss2 * weights_pred).mean()

        tot_loss = loss1 + loss2
        train_loss += tot_loss.item()

        losses.update(tot_loss.item(), imgs.size(0))
        predicates = torch.sigmoid(predicates)
        metric.update(predicates, targets_predicates)

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        # show information
        if (i+1) % opt.log_interval == 0:
            avg_loss = train_loss / opt.log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, losses.count, len(loader.dataset), 100. * (i + 1) / len(loader), avg_loss))
            train_loss = 0.0

    # show information
    recall = metric.compute_metrics()
    print('Train set ({:d} samples): Average loss: {:.4f}\tRecall: {:.4f}'.format(
        losses.count, losses.avg, recall))

    return losses.avg, recall
