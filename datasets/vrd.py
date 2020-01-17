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


<<<<<<< HEAD
class VrdDataset(Dataset):
    """VRD dataset."""

    def __init__(self, dataset_path, type, transform=None):
        # read annotations file
        with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
            self.annotations = json.load(f)

        # read image filenames
        with open(os.path.join(dataset_path, f'{type}.txt'), 'r') as f:
            image_names = f.read()

        self.imgs_list = image_names.split('\n')[:-1]

        self.transform = transform
        self.root = os.path.join(
            dataset_path, 'sg_dataset', f'sg_{type}_images')

    def __len__(self):
        return len(self.imgs_list)

    def pre_process(self, img, annotation):
        # list containing cropped unioned images
        cropped_imgs = []
        spatial_locations = []
        word_vectors = []
        predicate_list = []
        for sub_pred_obj in annotation:
            bbox_sub = sub_pred_obj['subject']['bbox']
            bbox_obj = sub_pred_obj['object']['bbox']
            # convert to x1,y1,x2,y2
            x1_sub, y1_sub, x2_sub, y2_sub = bbox_sub[2], bbox_sub[0], bbox_sub[3], bbox_sub[1]
            x1_obj, y1_obj, x2_obj, y2_obj = [
                bbox_obj[2], bbox_obj[0], bbox_obj[3], bbox_obj[1]]
            # get subject, object category
            sub_category = sub_pred_obj['subject']['category']
            object_category = sub_pred_obj['object']['category']
            # takes union
            polygons = [box(x1_sub, y1_sub, x2_sub, y2_sub),
                        box(x1_obj, y1_obj, x2_obj, y2_obj)]
            unioned = cascaded_union(polygons)
            unioned = unioned.bounds
            x1_unioned, y1_unioned, x2_unioned, y2_unioned = unioned
            # crop image
            cropped_img = img.crop((int(x1_unioned), int(
                y1_unioned), int(x2_unioned), int(y2_unioned)))
            img_w, img_h = cropped_img.size

            cropped_img = self.transform(cropped_img)

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
            bbox_sub_scaled = [sub_x1//factor_w, sub_y1 //
                               factor_h, sub_x2//factor_w, sub_y2//factor_h]
            bbox_obj_scaled = [obj_x1//factor_w, obj_y1 //
                               factor_h, obj_x2//factor_w, obj_y2//factor_h]

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
            word_vectors.append([sub_category, object_category])
            # predicate label
            predicate = sub_pred_obj['predicate']
            predicate_list.append(predicate)

        imgs = torch.stack(cropped_imgs)
        spatial_locations = torch.Tensor(spatial_locations)
        word_vectors = torch.Tensor(word_vectors)
        targets = torch.Tensor(predicate_list)
        return imgs, spatial_locations, word_vectors, targets

    def my_collate(self, batch):
        imgs = []
        spatial_locations = []
        word_vectors = []
        targets = []
        for item in batch:
            # remove incomplete annotations
            if (len(item[0].shape) == 4):
                imgs.append(item[0])
                spatial_locations.append(item[1])
                word_vectors.append(item[2])
                targets.append(item[3])

        imgs = torch.cat(imgs)
        spatial_locations = torch.cat(spatial_locations)
        word_vectors = torch.cat(word_vectors)
        word_vectors = word_vectors.type(torch.LongTensor)
        targets = torch.cat(targets)
        # flatten
        targets = targets.view(-1)
        targets = targets.type(torch.LongTensor)
        return imgs, spatial_locations, word_vectors, targets

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs_list[idx])
        img = Image.open(img_path)

        annotation = self.annotations[self.imgs_list[idx]]
        imgs, spatial_locations, word_vectors, targets = self.pre_process(
            img, annotation)
        return (imgs, spatial_locations, word_vectors, targets)

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)
=======
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

	def prepare_data(self, img, annotation, detection):
		cropped_imgs = []
		spatial_locations = []
		word_vectors = []
		predicate_list = []
		binary_targets = []

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
					binary_targets.append(1)
				else:
					predicate = [0 for _ in range(self.num_classes)]
					predicate_list.append(predicate)
					binary_targets.append(0)


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

				# prepare word vectors
				word_vectors.append([sub_label_gt, obj_label_gt])
				
				# prepare predicates
				predicate = sub_pred_obj['predicate']
				predicate = one_hot_encode(predicate, self.num_classes)
				predicate_list.append(predicate)
				binary_targets.append(1)
			

		imgs = torch.stack(cropped_imgs)
		spatial_locations = torch.Tensor(spatial_locations)
		word_vectors = torch.Tensor(word_vectors)
		predicates = torch.Tensor(predicate_list)
		binary_targets = torch.Tensor(binary_targets)
		return imgs, spatial_locations, word_vectors, predicates, binary_targets

	def my_collate(self, batch):
		imgs = []
		spatial_locations = []
		word_vectors = []
		predicates = []
		binary_targets = []
		for item in batch:
			# remove incomplete annotations
			if (len(item[0].shape) == 4):
				imgs.append(item[0])
				spatial_locations.append(item[1])
				word_vectors.append(item[2])
				predicates.append(item[3])
				binary_targets.append(item[4])

		imgs = torch.cat(imgs)
		spatial_locations = torch.cat(spatial_locations)
		word_vectors = torch.cat(word_vectors)
		word_vectors = word_vectors.type(torch.LongTensor)
		predicates = torch.cat(predicates)
		binary_targets = torch.cat(binary_targets)

		# flatten
		# targets = targets.view(-1)
		# targets = targets.type(torch.LongTensor)
		binary_targets = binary_targets.view(-1,1)
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

	
>>>>>>> v2
