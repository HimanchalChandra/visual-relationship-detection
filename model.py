import torch
from torch import nn
from models import resnet, vgg, vs_sw


def generate_model(opt):
    # assert opt.model in [
    #     'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    # ]
    if opt.model == 'resnet':
        model = resnet.Net(num_classes=70)  
    elif opt.model == 'vgg':
        model = vgg.Net(num_classes=70) 
    elif opt.model == 'vs_sw':
        model = vs_sw.Net(num_classes=70) 

        
    return model, model.parameters()
