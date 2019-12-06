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
from utils.util import calc_iou, calc_intersection, Scale


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
        self.scale = Scale()
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

            # feature scaling
            bbox_sub_scaled = self.scale(bbox_sub_scaled)
            bbox_obj_scaled = self.scale(bbox_obj_scaled)

            spatial_locations.append([iou, bbox_sub_scaled[0].item(), bbox_sub_scaled[1].item(), bbox_sub_scaled[2].item(), bbox_sub_scaled[3].item(),
                                      bbox_obj_scaled[0].item(), bbox_obj_scaled[1].item(), bbox_obj_scaled[2].item(), bbox_obj_scaled[3].item(), cflag_sub, cflag_obj])
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
