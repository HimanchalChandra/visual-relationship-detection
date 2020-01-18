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
from models import MFURLN
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
	model = MFURLN(num_classes=70)
	model = model.to(device)

	model = nn.DataParallel(model)


	# load pretrained weights
	checkpoint = torch.load('./snapshots/model26.pth')
	model.load_state_dict(checkpoint['model_state_dict'])
	print("Model Restored")

	for img_name in os.listdir('images'):
		# try:
			img_rgb = Image.open(f'./images/sf.jpg')
			img_bgr = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
			img_bgr = Image.fromarray(img_bgr)

			detections = retina_net.detect(img_bgr)
			print(detections)
			cropped_imgs = []
			spatial_locations = []
			spatial_locations1 = []
			word_vectors = []
			word_names = []
			detections1 = detections.copy()
			j=0
			for sub_label, x1_sub, y1_sub, x2_sub, y2_sub in detections:
				for obj_label, x1_obj, y1_obj, x2_obj, y2_obj in detections1:
					if ([sub_label, x1_sub, y1_sub, x2_sub, y2_sub] == [obj_label, x1_obj, y1_obj, x2_obj, y2_obj]):
						continue
					
					# takes union of sub and obj
					polygons = [box(x1_sub, y1_sub, x2_sub, y2_sub),
									box(x1_obj, y1_sub, x2_sub, y2_sub)]
					unioned = cascaded_union(polygons)
					unioned = unioned.bounds
					xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned
					# crop image
					cropped_img = img_rgb.crop((int(xmin_unioned), int(
						ymin_unioned), int(xmax_unioned), int(ymax_unioned)))
					cropped_img = transform(cropped_img)
					cropped_imgs.append(cropped_img)

					# prepare  spatial locations
					sub_xmin = x1_sub
					sub_ymin = y1_sub
					sub_xmax = x2_sub
					sub_ymax = y2_sub
					obj_xmin = x1_obj
					obj_ymin = y1_obj
					obj_xmax = x2_obj
					obj_ymax = y2_obj

					sub_x1 = int((sub_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
					sub_y1 = int((sub_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
					sub_x2 = int((sub_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
					sub_y2 = int((sub_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

					obj_x1 = int((obj_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
					obj_y1 = int((obj_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
					obj_x2 = int((obj_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
					obj_y2 = int((obj_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

					spatial_locations.append([sub_x1, sub_y1, sub_x2, sub_y2, obj_x1, obj_y1, obj_x2, obj_y2])
					spatial_locations1.append([sub_xmin, sub_ymin, sub_xmax, sub_ymax, obj_xmin, obj_ymin, obj_xmax, obj_ymax])

					# prepare word vectors
					word_vectors.append([word2int_obj[sub_label], word2int_obj[obj_label]])
			
			imgs = torch.stack(cropped_imgs)
			spatial_locations = torch.Tensor(spatial_locations)
			spatial_locations1 = torch.Tensor(spatial_locations1)
			word_vectors = torch.Tensor(word_vectors)
			word_vectors = word_vectors.type(torch.LongTensor)
		
			print(imgs.shape)
			print(spatial_locations.shape)
			print(word_vectors.shape)

			with torch.no_grad():
				confidences, predicates = model(imgs, spatial_locations, word_vectors)
			confidences = torch.sigmoid(confidences)
			predicates = torch.sigmoid(predicates)

			print("confidences")
			print(confidences.t())

			scores, preds = predicates.max(dim=1, keepdim=True) # get the index of the max log-probability

			print("prdicate scores")
			print(scores.t())

			

			# apply mask for thresholding
			# mask = scores > 0.2
			mask = scores > 0.1
			preds = preds[mask]
			scores = scores[mask]

			mask1 = torch.cat([mask,mask], dim=1)
			word_vectors = word_vectors[mask1]
			word_vectors = word_vectors.view(-1,2)

			mask2 = torch.cat([mask,mask,mask,mask,mask,mask,mask,mask], dim=1)
			spatial_locations1 = spatial_locations1[mask2]
			spatial_locations1 = spatial_locations1.view(-1,8)

			# mask2 = torch.cat([mask,mask,mask,mask,mask,mask,mask,mask,mask,mask,mask], dim=1)
			# spatial_locations = spatial_locations[mask2]
			# spatial_locations = spatial_locations.view(-1,11)

			draw = img_bgr.copy()
			draw = np.array(draw)

			# # save results
			# for k,img in enumerate(imgs):
			# 	img = transforms.ToPILImage()(img)
			# 	score, pred = outputs[k].max(dim=0, keepdim=True) # get the index of the max log-probability
			# 	print(score)
			# 	if (score.item() > 0.95):
			# 		bboxes = spatial_locations1[k]
			# 		draw1 = ImageDraw.Draw(img)
			# 		draw1.rectangle(((int(bboxes[1].item()), int(bboxes[2].item())), (int(bboxes[3].item()), int(bboxes[4].item()))))
			# 		draw1.rectangle(((int(bboxes[5].item()), int(bboxes[6].item())), (int(bboxes[7].item()), int(bboxes[8].item()))))
			# 		print(int(bboxes[1].item()), int(bboxes[2].item()), int(bboxes[3].item()), int(bboxes[4].item()))
			# 		print(int(bboxes[5].item()), int(bboxes[6].item()), int(bboxes[7].item()), int(bboxes[8].item()))
			# 		#cv2.rectangle(img, (int(sub_obj[0].item()), int(sub_obj[1].item())), (int(sub_obj[2].item()), int(sub_obj[3].item())), (255,0,0), 2)
			# 		img.save(f'results/{str(j)}.jpg')
			# 		j+=1


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
				bboxes = spatial_locations1[i]
				centr_sub = ( int((bboxes[0].item()+ bboxes[2].item())/2) , int((bboxes[1].item()+ bboxes[3].item())/2) )
				centr_obj = ( int((bboxes[4].item()+ bboxes[6].item())/2) , int((bboxes[5].item()+ bboxes[7].item())/2) )

				lineThickness = 1
				cv2.line(draw, centr_sub, centr_obj, (0,255,0), lineThickness)
				print(f'{i}) {int2word_obj[word_vectors[i][0].item()]} {int2word_pred[pred.item()]} {int2word_obj[word_vectors[i][1].item()]} ,score:{scores[i].item()}')
				
				# if (i==5):
				# 	break

				predicate_point = ( int((centr_sub[0] + centr_obj[0])/2 ) , int((centr_sub[1] + centr_obj[1])/2 ))

				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(draw, int2word_pred[pred.item()], predicate_point, font, .5,(255,255,255),1,cv2.LINE_AA)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(draw, int2word_obj[word_vectors[i][0].item()], centr_sub, font, .5,(255,255,255),1,cv2.LINE_AA)
				cv2.putText(draw, int2word_obj[word_vectors[i][1].item()], centr_obj, font, .5,(255,255,255),1,cv2.LINE_AA)

				# cv2.putText(draw, int2word_obj[word_vectors[i][0].item()], centr_sub, font, .5,(255,255,255),1,cv2.LINE_AA)
				# cv2.putText(draw, int2word_obj[word_vectors[i][1].item()], centr_obj, font, .5,(255,255,255),1,cv2.LINE_AA)

			
			cv2.imwrite(f'./outputs/{img_name}', draw)
		# except:
		# 	continue
		
			
	

			
if __name__ == "__main__":
	main()