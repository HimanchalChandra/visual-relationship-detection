import torch
from torch import nn
from models import MFURLN


def generate_model(opt):
    # assert opt.model in [
    #     'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    # ]
    # if opt.model == 'resnet':
    #     model = resnet.Net(opt.num_classes)  
    # elif opt.model == 'vgg':
    #     model = vgg.Net(opt.num_classes) 
    # elif opt.model == 'vs_sw':
    #     model = vs_sw.Net(opt.num_classes) 
    if opt.model == 'mfurln':
        model = MFURLN(opt.num_classes) 

    return model, model.parameters()
