import numpy as np
import cv2
from utils.util import calc_intersection,calc_iou
b1 = [25,25,25,25]


img = np.ones((480,640,3))


cv2.rectangle(img, (70, 70), (200, 200), (255,0,0), 2)
cv2.rectangle(img, (210, 210), (300, 300), (255,0,0), 2)

rect1center = ((int(70+200/2)), (int(70+200/2)))

print(rect1center)

cv2.line(img, rect1center, rect1center, (255,0,0), 2)

# print(calc_intersection([70, 70, 200, 200], [210, 210, 300, 300] ))

cv2.imshow('window', img)
cv2.waitKey(0)

# import torch
# from torchvision import models
# import torch.nn as nn
# from collections import OrderedDict

# model = models.resnet50(pretrained=True)
# modules = list(model.children())[:-1]

# modules += [nn.Linear(model.fc.in_features, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True)]
# modules += [nn.Linear(4096, 70)]

# model = nn.Sequential(*modules)

# print(model)
