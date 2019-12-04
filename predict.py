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
from dataset import VrdDataset
import json
import torch.nn as nn

import argparse
import random
import cv2
from models.resnet import Net
from shapely.geometry import box
from shapely.ops import cascaded_union
from retinanet.obj_det_retinanet import ObjDetRetinanet
from utils.util import calc_iou,calc_intersection
from opts import parse_opts
# model = vgg16(pretrained=True)
# modules = list(model.children())[:-2]  
# base_net = nn.Sequential(*modules)

# print(base_net)
# import torch

opt = parse_opts()

retina_net = ObjDetRetinanet()
		
with open(os.path.join(opt.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
	predicates = json.load(f)

with open(os.path.join(opt.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
    objects = json.load(f)

word2int_obj = {}
for i, obj in enumerate(objects):
	word2int_obj[obj] = i

int2word_obj = {}
for i, obj in enumerate(objects):
	int2word_obj[i] = obj

int2word_pred = {}
for i, predicate in enumerate(predicates):
	int2word_pred[i] = predicate


			
def main():


	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")

	
	transform = transforms.Compose([
							#transforms.RandomHorizontalFlip(),
							#transforms.RandomRotation(10),
							#transforms.RandomCrop(32, padding=3),
							transforms.Resize((224,224)),
							transforms.ToTensor(),
							transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
						])



	# define model
	model = Net(num_classes=70)
	model = model.to(device)


	# load pretrained weights
	checkpoint = torch.load('./snapshots/model305.pth', map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])
	print("Model Restored")


	img = Image.open('/Users/pranoyr/Desktop/test/8559246586_4bd43f9505_b.jpg')
	detections = retina_net.detect(img)
	print(detections)
	cropped_imgs = []
	spatial_locations = []
	word_vectors = []
	word_names = []
	detections1 = detections.copy()
	j=0
	for sub_label, x1_sub, y1_sub, x2_sub, y2_sub in detections:
		for obj_label, x1_obj, y1_obj, x2_obj, y2_obj in detections1:
			if ([sub_label, x1_sub, y1_sub, x2_sub, y2_sub] == [obj_label, x1_obj, y1_obj, x2_obj, y2_obj]):
				continue
			
			# takes union
			polygons = [box(x1_sub, y1_sub, x2_sub, y2_sub),
							box(x1_obj, y1_obj, x2_obj, y2_obj)]
			unioned = cascaded_union(polygons)
			unioned = unioned.bounds
			x1_unioned, y1_unioned, x2_unioned, y2_unioned = unioned
			# crop image
			cropped_img = img.crop((int(x1_unioned), int(y1_unioned), int(x2_unioned), int(y2_unioned))) 

			# cv2.imshow('window',np.array(cropped_img))
			# cv2.waitKey(0)

			img_w, img_h = cropped_img.size

			cropped_img = transform(cropped_img)
		
			factor_h = img_h/224
			factor_w = img_w/224

			cropped_imgs.append(cropped_img)
			# spatial locations
			# find bounding box coordinates relative to unioned image
			sub_x1 = x1_sub - int(x1_unioned)
			sub_y1 = y1_sub - int(y1_unioned)
			sub_x2 = x2_sub - int(x1_unioned)
			sub_y2 = y2_sub - int(y1_unioned)

			obj_x1 = x1_obj - int(x1_unioned)
			obj_y1 = y1_obj - int(y1_unioned)
			obj_x2 = x2_obj - int(x1_unioned)
			obj_y2 = y2_obj - int(y1_unioned)
			
			# rescaling of bboxes for image with dim (224,224)
			bbox_sub_scaled = [sub_x1//factor_w, sub_y1//factor_h, sub_x2//factor_w, sub_y2//factor_h]
			bbox_obj_scaled = [obj_x1//factor_w, obj_y1//factor_h, obj_x2//factor_w, obj_y2//factor_h]

			# calculate iou
			iou = calc_iou(bbox_sub_scaled, bbox_obj_scaled)

			# setting cflag for subject
			if calc_intersection(bbox_obj_scaled, bbox_sub_scaled) == 1:
				cflag_sub = 1
			else:
				cflag_sub = 0

			# setting cflag for object
			if calc_intersection(bbox_sub_scaled, bbox_obj_scaled) == 1:
				cflag_obj = 1
			else:
				cflag_obj = 0

			spatial_locations.append([iou, bbox_sub_scaled[0], bbox_sub_scaled[1], bbox_sub_scaled[2], bbox_sub_scaled[3],
									  bbox_obj_scaled[0], bbox_obj_scaled[1], bbox_obj_scaled[2], bbox_obj_scaled[3], cflag_sub, cflag_obj])
			# word vectors
			word_vectors.append([word2int_obj[sub_label], word2int_obj[obj_label]])
		
		

	imgs = torch.stack(cropped_imgs)
	spatial_locations = torch.Tensor(spatial_locations)
	word_vectors = torch.Tensor(word_vectors)
	word_vectors = word_vectors.type(torch.LongTensor)
   
	print(imgs.shape)
	print(spatial_locations.shape)
	print(word_vectors.shape)

	

	outputs = model(imgs, spatial_locations, word_vectors)

	outputs = torch.softmax(outputs, dim=1)
	scores, preds = outputs.max(dim=1, keepdim=True) # get the index of the max log-probability

	# apply mask for thresholding
	mask = scores > 0.7

	preds = preds[mask]
	scores = scores[mask]
	mask1 = torch.cat([mask,mask], dim=1)
	word_vectors = word_vectors[mask1]
	word_vectors = word_vectors.view(-1,2)

	# mask2 = torch.cat([mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask], dim=1)
	# spatial_locations = spatial_locations[mask2]
	# spatial_locations = spatial_locations.view(-1,11)


	# save results
	for k,img in enumerate(imgs):
		img = transforms.ToPILImage()(img)
		score, pred = outputs[k].max(dim=0, keepdim=True) # get the index of the max log-probability
		print(score)
		if (score.item() > 0.7):
			bboxes = spatial_locations[k]
			draw1 = ImageDraw.Draw(img)
			draw1.rectangle(((int(bboxes[1].item()), int(bboxes[2].item())), (int(bboxes[3].item()), int(bboxes[4].item()))))
			draw1.rectangle(((int(bboxes[5].item()), int(bboxes[6].item())), (int(bboxes[7].item()), int(bboxes[8].item()))))
		
			#cv2.rectangle(img, (int(sub_obj[0].item()), int(sub_obj[1].item())), (int(sub_obj[2].item()), int(sub_obj[3].item())), (255,0,0), 2)
			img.save(f'results/{str(j)}.jpg')
			j+=1


	# # save results
	# for k,img in enumerate(imgs):
	# 	img = transforms.ToPILImage()(img)
	# 	sub_obj = spatial_locations[k]
	# 	draw1 = ImageDraw.Draw(img)
	# 	print(sub_obj)
	# 	draw1.rectangle(((int(sub_obj[1].item()), int(sub_obj[2].item())), (int(sub_obj[3].item()), int(sub_obj[4].item()))))
	# 	draw1.rectangle(((int(sub_obj[5].item()), int(sub_obj[6].item())), (int(sub_obj[7].item()), int(sub_obj[8].item()))))
	# 	img.save(f'results/{str(j)}.jpg')
	# 	j+=1

	for i, pred in enumerate(preds):
		print(f'{i}) {int2word_obj[word_vectors[i][0].item()]} {int2word_pred[pred.item()]} {int2word_obj[word_vectors[i][1].item()]} ,score:{scores[i].item()}')
		
   
	
  

		
if __name__ == "__main__":
	main()