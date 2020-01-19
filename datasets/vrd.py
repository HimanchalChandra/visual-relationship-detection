import numpy as np
import cv2
from shapely.geometry import box
from shapely.ops import cascaded_union
from PIL import Image
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import json
from utils.util import calc_iou, calc_intersection


def one_hot_encode(integer_encoding, num_classes):
	""" One hot encode.
	"""
	onehot_encoded = [0 for _ in range(num_classes)]
	onehot_encoded[integer_encoding] = 1
	return onehot_encoded

def make_image_list(dataset_path, type):
	imgs_list = []
	with open(os.path.join(dataset_path, 'json_dataset',f'annotations_{type}.json'), 'r') as f:
		annotations = json.load(f)
	sg_images = os.listdir(os.path.join(dataset_path,'sg_dataset',f'sg_{type}_images'))

	annotations_copy = annotations.copy()
	for ann in annotations.items():
		if(not annotations[ann[0]] or ann[0] not in sg_images):
			annotations_copy.pop(ann[0])
 
	for ann in annotations_copy.items():
		imgs_list.append(ann[0])
	return imgs_list


class VrdDataset(Dataset):
	"""VRD dataset."""

	def __init__(self, dataset_path, num_classes, type, transform=None):
		# read annotations file
		with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
			self.annotations = json.load(f)

		with open(os.path.join(dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
			objects = json.load(f)

		with open(os.path.join(dataset_path, 'json_dataset', 'detections.json'), 'r') as f:
			self.detections = json.load(f)
		
		# read image filenames
		with open(os.path.join(dataset_path, f'{type}.txt'), 'r') as f:
			image_names = f.read()
		self.imgs_list = image_names.split('\n')[:-1]

		self.class2idx_obj = {}
		for i, obj in enumerate(objects):
			self.class2idx_obj[obj] = i
			
		# self.imgs_list = make_image_list(dataset_path, type)
		self.num_classes = num_classes
		self.transform = transform
		self.root = os.path.join(
			dataset_path, 'sg_dataset', f'sg_{type}_images')

	def __len__(self):
		return len(self.imgs_list)

	# def prepare_rois(self, sub_bbox, obj_bbox, unioned, factor_h, factor_w):
	# 	xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned

	# 	sub_xmin = sub_bbox[0]
	# 	sub_ymin = sub_bbox[1]
	# 	sub_xmax = sub_bbox[2]
	# 	sub_ymax = sub_bbox[3]
	# 	obj_xmin = obj_bbox[0]
	# 	obj_ymin = obj_bbox[1]
	# 	obj_xmax = obj_bbox[2]
	# 	obj_ymax = obj_bbox[3]

	# 	# find bounding box coordinates relative to unioned image
	# 	sub_x1 = sub_xmin - int(xmin_unioned)
	# 	sub_y1 = sub_ymin - int(ymin_unioned)
	# 	sub_x2 = sub_xmax - int(xmin_unioned)
	# 	sub_y2 = sub_ymax - int(ymin_unioned)

	# 	obj_x1 = obj_xmin - int(xmin_unioned)
	# 	obj_y1 = obj_ymin - int(ymin_unioned)
	# 	obj_x2 = obj_xmax - int(xmin_unioned)
	# 	obj_y2 = obj_ymax - int(ymin_unioned)

	# 	# rescaling of bboxes for image with dim (224,224)
	# 	bbox_sub_scaled = [sub_x1//factor_w, sub_y1 //
	# 						factor_h, sub_x2//factor_w, sub_y2//factor_h]
	# 	bbox_obj_scaled = [obj_x1//factor_w, obj_y1 //
	# 						factor_h, obj_x2//factor_w, obj_y2//factor_h]

	# 	rois = {'sub': torch.Tensor([bbox_sub_scaled]), 'obj': torch.Tensor([bbox_obj_scaled])}
	# 	return rois


	def prepare_data(self, img, annotation, detection):
		cropped_imgs = []
		spatial_locations = []
		word_vectors = []
		predicate_list = []
		binary_targets = []
		# rois_sub = []
		# rois_obj = []

		detection1 = detection.copy()
		for sub in detection:
			for obj in detection1:
				sub_label = self.class2idx_obj[sub['label']]
				obj_label = self.class2idx_obj[obj['label']]
				sub_bbox = [int(sub['bbox'][0]), int(sub['bbox'][1]), int(sub['bbox'][2]), int(sub['bbox'][3])]
				obj_bbox = [int(obj['bbox'][0]), int(obj['bbox'][1]), int(obj['bbox'][2]), int(obj['bbox'][3])]

				if sub_label == obj_label and sub_bbox == obj_bbox:
					continue
				# takes union of sub and obj
				polygons = [box(sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3]),
								box(obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3])]
				unioned = cascaded_union(polygons)
				unioned = unioned.bounds
				xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned
				# crop image
				cropped_img = img.crop((int(xmin_unioned), int(
					ymin_unioned), int(xmax_unioned), int(ymax_unioned)))

				img_w, img_h = cropped_img.size
				factor_h = img_h/224
				factor_w = img_w/224

				cropped_img = self.transform(cropped_img)
				cropped_imgs.append(cropped_img)

				# prepare  spatial locations
				sub_xmin = sub_bbox[0]
				sub_ymin = sub_bbox[1]
				sub_xmax = sub_bbox[2]
				sub_ymax = sub_bbox[3]
				obj_xmin = obj_bbox[0]
				obj_ymin = obj_bbox[1]
				obj_xmax = obj_bbox[2]
				obj_ymax = obj_bbox[3]

				sub_x1 = int((sub_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
				sub_y1 = int((sub_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
				sub_x2 = int((sub_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
				sub_y2 = int((sub_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

				obj_x1 = int((obj_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
				obj_y1 = int((obj_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
				obj_x2 = int((obj_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
				obj_y2 = int((obj_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

				spatial_locations.append([sub_x1, sub_y1, sub_x2, sub_y2, obj_x1, obj_y1, obj_x2, obj_y2])

				# # prepare rois
				# rois = self.prepare_rois(sub_bbox, obj_bbox, unioned, factor_h, factor_w)
				# rois_sub.append(rois['sub'])
				# rois_obj.append(rois['obj'])

				# prepare word vectors
				word_vectors.append([sub_label, obj_label])

				data_type = 'undetermined'
				for sub_pred_obj in annotation:
					sub_label_gt = sub_pred_obj['subject']['category']
					obj_label_gt = sub_pred_obj['object']['category']
					sub_bbox_gt = sub_pred_obj['subject']['bbox']
					obj_bbox_gt = sub_pred_obj['object']['bbox']
					# convert to x1, y1, x2, y2
					sub_bbox_gt = [sub_bbox_gt[2], sub_bbox_gt[0], sub_bbox_gt[3], sub_bbox_gt[1]]
					obj_bbox_gt = [obj_bbox_gt[2], obj_bbox_gt[0], obj_bbox_gt[3], obj_bbox_gt[1]]

					# calculates iou of detections and gt for sub and obj.
					sub_iou = calc_iou(sub_bbox, sub_bbox_gt)
					obj_ioc = calc_iou(obj_bbox, obj_bbox_gt)

					# prepare predicates for determined and undetermined data
					if (sub_label == sub_label_gt and obj_label == obj_label_gt and sub_iou > 0.5 and obj_ioc > 0.5):
						data_type = 'determined'
						break
				
				if (data_type == 'determined'):
					predicate = sub_pred_obj['predicate']
					predicate = one_hot_encode(predicate, self.num_classes)
					predicate_list.append(predicate)
					binary_targets.append(one_hot_encode(1, num_classes = 2))
				else:
					# predicate = [0 for _ in range(self.num_classes)]
					predicate = one_hot_encode(70, self.num_classes)
					predicate_list.append(predicate)
					binary_targets.append(one_hot_encode(0, num_classes = 2))


		if not cropped_imgs:
			for sub_pred_obj in annotation:
				sub_label_gt = sub_pred_obj['subject']['category']
				obj_label_gt = sub_pred_obj['object']['category']
				sub_bbox_gt = sub_pred_obj['subject']['bbox']
				obj_bbox_gt = sub_pred_obj['object']['bbox']
				# convert to x1, y1, x2, y2
				sub_bbox_gt = [sub_bbox_gt[2], sub_bbox_gt[0], sub_bbox_gt[3], sub_bbox_gt[1]]
				obj_bbox_gt = [obj_bbox_gt[2], obj_bbox_gt[0], obj_bbox_gt[3], obj_bbox_gt[1]]

				# takes union of sub and obj
				polygons = [box(sub_bbox_gt[0], sub_bbox_gt[1], sub_bbox_gt[2], sub_bbox_gt[3]),
								box(obj_bbox_gt[0], obj_bbox_gt[1], obj_bbox_gt[2], obj_bbox_gt[3])]
				unioned = cascaded_union(polygons)
				unioned = unioned.bounds
				xmin_unioned, ymin_unioned, xmax_unioned, ymax_unioned = unioned
				# crop image
				cropped_img = img.crop((int(xmin_unioned), int(
					ymin_unioned), int(xmax_unioned), int(ymax_unioned)))

				img_w, img_h = cropped_img.size
				factor_h = img_h/224
				factor_w = img_w/224

				cropped_img = self.transform(cropped_img)
				cropped_imgs.append(cropped_img)

				# prepare  spatial locations
				sub_xmin = sub_bbox_gt[0]
				sub_ymin = sub_bbox_gt[1]
				sub_xmax = sub_bbox_gt[2]
				sub_ymax = sub_bbox_gt[3]
				obj_xmin = obj_bbox_gt[0]
				obj_ymin = obj_bbox_gt[1]
				obj_xmax = obj_bbox_gt[2]
				obj_ymax = obj_bbox_gt[3]

				sub_x1 = int((sub_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
				sub_y1 = int((sub_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
				sub_x2 = int((sub_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
				sub_y2 = int((sub_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

				obj_x1 = int((obj_xmin - xmin_unioned)/(xmax_unioned - xmin_unioned))
				obj_y1 = int((obj_ymin - ymin_unioned)/(ymax_unioned - ymin_unioned))
				obj_x2 = int((obj_xmax - xmax_unioned)/(xmax_unioned - xmin_unioned))
				obj_y2 = int((obj_ymax - ymax_unioned)/(ymax_unioned - ymin_unioned))

				spatial_locations.append([sub_x1, sub_y1, sub_x2, sub_y2, obj_x1, obj_y1, obj_x2, obj_y2])

				# # prepare rois
				# rois = self.prepare_rois(sub_bbox, obj_bbox, unioned, factor_h, factor_w)
				# rois_sub.append(rois['sub'])
				# rois_obj.append(rois['obj'])

				# prepare word vectors
				word_vectors.append([sub_label_gt, obj_label_gt])
				
				# prepare predicates
				predicate = sub_pred_obj['predicate']
				predicate = one_hot_encode(predicate, self.num_classes)
				predicate_list.append(predicate)
				binary_targets.append(one_hot_encode(1, num_classes = 2))
			

		imgs = torch.stack(cropped_imgs)
		spatial_locations = torch.Tensor(spatial_locations)
		word_vectors = torch.Tensor(word_vectors)
		predicates = torch.Tensor(predicate_list)
		binary_targets = torch.Tensor(binary_targets)
		# rois_sub = torch.stack(rois_sub)
		# rois_obj = torch.stack(rois_obj)
		return imgs, spatial_locations, word_vectors, predicates, binary_targets

	def my_collate(self, batch):
		imgs = []
		spatial_locations = []
		word_vectors = []
		predicates = []
		binary_targets = []
		# rois_sub = []
		# rois_obj = []
		for item in batch:
			# remove incomplete annotations
			if (len(item[0].shape) == 4):
				imgs.append(item[0])
				spatial_locations.append(item[1])
				word_vectors.append(item[2])
				predicates.append(item[3])
				binary_targets.append(item[4])
				# rois_sub.append(item[5])
				# rois_obj.append(item[6])
				

		imgs = torch.cat(imgs)
		spatial_locations = torch.cat(spatial_locations)
		word_vectors = torch.cat(word_vectors)
		word_vectors = word_vectors.type(torch.LongTensor)
		predicates = torch.cat(predicates)
		binary_targets = torch.cat(binary_targets)
		# rois_sub = torch.cat(rois_sub)
		# rois_obj = torch.cat(rois_obj)

		# flatten
		# targets = targets.view(-1)
		# targets = targets.type(torch.LongTensor)
		#binary_targets = binary_targets.view(-1,1)
		return imgs, spatial_locations, word_vectors, predicates, binary_targets

	def __getitem__(self, idx):
		img_path = os.path.join(self.root, self.imgs_list[idx])
		img = Image.open(img_path)

		# load annotated data
		annotation = self.annotations[self.imgs_list[idx]]
		# load detections from object detection output
		detection  = self.detections[self.imgs_list[idx]]

		# prepare determined and undetermined batches
		imgs, spatial_locations, word_vectors, predicates, binary_targets = self.prepare_data(
			img, annotation, detection)

		return (imgs, spatial_locations, word_vectors, predicates, binary_targets)

	