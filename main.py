import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torchvision.models import vgg16
from PIL import Image, ImageFont, ImageDraw
from torch.nn import BCEWithLogitsLoss
from dataset import get_dataset
import json
import torch.nn as nn
from torchvision.models import resnet50
import tensorboardX
import argparse
from torch.optim import lr_scheduler
import random
import cv2
from training import train
from util import Logger
from validation import validate
from opts import parse_opts
from model import generate_model


# model = vgg16(pretrained=True)
# modules = list(model.children())[:-2]
# base_net = nn.Sequential(*modules)

# print(base_net)
# import torch


def main():
	opt = parse_opts()
	print(opt)

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{opt.gpu}" if use_cuda else "cpu")

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	train_transform = transforms.Compose([
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((224, 224)),
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])
	test_transform = transforms.Compose([
		# transforms.RandomHorizontalFlip(),
		# transforms.RandomRotation(10),
		#transforms.RandomCrop(32, padding=3),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
			0.229, 0.224, 0.225])
	])

	# data loaders
	train_dataset = get_dataset(opt, 'train', transform=train_transform)
	train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
							  num_workers=0, collate_fn=train_dataset.my_collate)
	val_dataset = get_dataset(opt, 'test', transform=test_transform)
	val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
							num_workers=0, collate_fn=val_dataset.my_collate)

	print(f'Number of training examples: {len(train_loader.dataset)}')
	print(f'Number of validation examples: {len(val_loader.dataset)}')

	# define model
	model, parameters = generate_model(opt)
	# model = model.to(device)
	model = model.cuda()

	if torch.cuda.device_count() > 1:
	  	print("Let's use", torch.cuda.device_count(), "GPUs!")
  		model = nn.DataParallel(model)

	if opt.nesterov:
		dampening = 0
	else:
		dampening = opt.dampening
	# define optimizer and criterion
	# optimizer = optim.Adam(parameters)
	optimizer = optim.SGD(
			model.parameters(),
			lr=opt.learning_rate,
			momentum=opt.momentum,
			dampening=dampening,
			weight_decay=opt.weight_decay,
			nesterov=opt.nesterov)
	# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
	drop_after_epoch = [10, 20, 30]
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.5)
	criterion = BCEWithLogitsLoss()

	# pretrained weights
	if opt.weights:
		checkpoint = torch.load(opt.weights)
		model.load_state_dict(checkpoint['model_state_dict'], strict=False)
		print("Pretrained weights loaded")

	# resume model, optimizer if already exists
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(epoch))
		start_epoch = epoch + 1
	else:
		start_epoch = 1

	# start training
	th = 10000
	for epoch in range(start_epoch, opt.epochs+1):
		# train, test model
		train_loss, train_recall = train(
			model, train_loader, criterion, optimizer, epoch, device, opt)
		scheduler.step()
		
		if (epoch) % opt.save_interval == 0:
			# val_loss, val_recall = validate(model, val_loader, criterion, epoch, device, opt)
			# scheduler.step(val_loss)
			# # write summary
			# summary_writer.add_scalar(
			#     'losses/train_loss', train_loss, global_step=epoch)
			# summary_writer.add_scalar(
			#     'losses/val_loss', val_loss, global_step=epoch)
			# summary_writer.add_scalar(
			#     'acc/train_acc', train_recall, global_step=epoch)
			# summary_writer.add_scalar(
			#     'acc/val_acc', val_recall, global_step=epoch)

			state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
					 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'model{epoch}.pth'))
			print("Epoch {} model saved!\n".format(epoch))


if __name__ == "__main__":
	main()
