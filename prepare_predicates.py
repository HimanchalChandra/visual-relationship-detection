import numpy as np

import cv2

from shapely.geometry import box
from shapely.ops import cascaded_union


# print(unioned.bounds)


# (324, 336), (458, 489), (255,0,0), 2)
# cv2.rectangle(img, (306, 94), (590, 175),

import os 
import json


with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/annotations_train.json', 'r') as f:
    train_file = json.load(f)

with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/objects.json', 'r') as f:
    objects = json.load(f)

with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/predicates.json', 'r') as f:
    predicates = json.load(f)

path = os.path.dirname(os.path.abspath(__file__))

object_list = []
classes = []
i = 0
# loop over annotations
for img_name, img_anns in train_file.items():

    img_path = '/Volumes/Seagate Expansion Drive/visual_genome/sg_dataset/sg_train_images/'+img_name
    # loop over sub,pred,obj of each annotation
    for sub_pred_obj in img_anns:
        # bbox for object
        bbox = sub_pred_obj['object']['bbox']
        name = objects[sub_pred_obj['object']['category']]
        
        #f.write(img_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',' + str(name))
        #f.write('\n')

        object_list.append(path + '/'+ img_name + ',' + str(bbox[2]) + ',' + str(bbox[0]) + ',' + str(bbox[3]) + ',' + str(bbox[1]) + ',' + str(name))
        # bboxes for subject
        bbox1 = sub_pred_obj['subject']['bbox']
        name = objects[sub_pred_obj['subject']['category']]
    
        predicate = predicates[sub_pred_obj['predicate']]

        
        classes.append(predicate)

        

        polygons = [box(bbox[2], bbox[0], bbox[3], bbox[1]),box(bbox1[2], bbox1[0], bbox1[3], bbox1[1])]
        unioned = cascaded_union(polygons)


        unioned = unioned.bounds
        

        img = cv2.imread(img_path)
        print(unioned)
        crop_img = img[int(unioned[1]):int(unioned[3]), int(unioned[0]):int(unioned[2])]
        print(crop_img.shape)
        cv2.imwrite('dataset/'+predicate+'/'+str(i)+'.jpg',crop_img)
        i+=1



        
# classes = set(classes)
# for i in classes:
#     os.mkdir('./dataset/'+i)