import json
import os
from opts import parse_opts


opt = parse_opts()

# prepare train.txt
with open(os.path.join(opt.dataset_path, 'json_dataset','annotations_train.json'), 'r') as f:
    annotations = json.load(f)
sg_train_images = os.listdir(os.path.join(opt.dataset_path,'sg_dataset','sg_train_images'))
sg_test_images = os.listdir(os.path.join(opt.dataset_path,'sg_dataset','sg_test_images'))

annotations_copy = annotations.copy()
for ann in annotations.items():
    if(not annotations[ann[0]] or ann[0] not in sg_train_images):
        annotations_copy.pop(ann[0])

with open(os.path.join(opt.dataset_path, 'train.txt'),'a') as f: 
    for ann in annotations_copy.items():
        f.write(ann[0])
        f.write('\n')

# prepare test.txt
with open(os.path.join(opt.dataset_path, 'json_dataset','annotations_test.json'), 'r') as f:
    annotations = json.load(f)

annotations_copy = annotations.copy()
for ann in annotations.items():
    if(not annotations[ann[0]] or ann[0] not in sg_test_images):
        annotations_copy.pop(ann[0])

with open(os.path.join(opt.dataset_path, 'test.txt'),'a') as f: 
    for ann in annotations_copy.items():
        f.write(ann[0])
        f.write('\n')

        



        

