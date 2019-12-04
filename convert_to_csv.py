import os 
import json

with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/annotations_train.json', 'r') as f:
    train_file = json.load(f)

with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/objects.json', 'r') as f:
    objects = json.load(f)

with open('/Volumes/Seagate Expansion Drive/visual_genome/json_dataset/predicates.json', 'r') as f:
    predicates = json.load(f)

path = os.path.dirname(os.path.abspath(__file__))

img_list = []
object_list = []
classes = []
# loop over annotations
for img_name, img_anns in train_file.items():
    # loop over sub,pred,obj of each annotation
    img_list.append(img_name)
    for sub_pred_obj in img_anns:
        # bbox for object
        bbox = sub_pred_obj['object']['bbox']
        name = objects[sub_pred_obj['object']['category']]
        classes.append(name)
        #f.write(img_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',' + str(name))
        #f.write('\n')

        object_list.append(path + '/'+ img_name + ',' + str(bbox[2]) + ',' + str(bbox[0]) + ',' + str(bbox[3]) + ',' + str(bbox[1]) + ',' + str(name))
        # bboxes for subject
        bbox = sub_pred_obj['subject']['bbox']
        name = objects[sub_pred_obj['subject']['category']]
        classes.append(name)
        
        object_list.append(path + '/'+ img_name + ',' + str(bbox[2]) + ',' + str(bbox[0]) + ',' + str(bbox[3]) + ',' + str(bbox[1]) + ',' + str(name))
        #f.write(img_name + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',' + str(name))
        #f.write('\n')

classes = set(classes)
with open('labels.csv','a') as f:
    for i, j in enumerate(classes):
        f.write(j+','+str(i))
        f.write('\n')


with open('annotations.csv','a') as f:
    object_list = set(object_list)
    for i in object_list:
        f.write(i)  
        f.write('\n')

with open('classes.txt','a') as f:
    for i in img_list:
        f.write(i)  
        f.write('\n')



        