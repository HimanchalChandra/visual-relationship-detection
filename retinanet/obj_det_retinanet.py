# import keras
import keras

# import keras_retinanet
from PIL import Image
from .keras_retinanet import models
from .keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from .keras_retinanet.utils.visualization import draw_box, draw_caption
from .keras_retinanet.utils.colors import label_color
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import csv
import time
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

path = os.path.dirname(os.path.abspath(__file__))

class ObjDetRetinanet:
    """ Class to detect objects using RetinaNet from an image batch.
    """

    def __init__(self):
        # use this environment flag to change which GPU to use
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        # set the modified tf session as backend in keras
        #keras.backend.tensorflow_backend.set_session(self.get_session())

        # adjust this to point to your downloaded/trained model
        # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
        model_path = os.path.join(path, "snapshots", "resnet50_vrd_inference.h5")

        # load retinanet model
        self.model = models.load_model(model_path, backbone_name="resnet50")

        # if the model is not converted to an inference model, use the line below
        # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
        # model = models.convert_model(model)

        # print(model.summary())

        # load label to names mapping for visualization purposes
    
        with open(os.path.join(path, 'labels.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            names_to_labels = {rows[0]: int(rows[1]) for rows in reader}

        self.labels_to_names = {v: k for k, v in names_to_labels.items()}

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def detect(self, img):
        """ Detect objects from from an image batch. 

        Args    
            image_batch : Image  of shape (None,None,None,None).
        Returns 
            det : 2D list of bounding boxes. 
        """
        # preprocess image for network
        img_rgb = np.array(img).copy()
        bgr_img = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
        bgr_img = preprocess_image(bgr_img)
        bgr_img, scale = resize_image(bgr_img)
        # create image batch
        image_batch = np.expand_dims(bgr_img, axis=0)

        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(image_batch)
        print("processing time: ", time.time() - start)
        # correct for image scale
        boxes /= scale
        # visualize detections
        
        results = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            b = box.astype(int)
            results.append([self.labels_to_names[label],b[0], b[1], b[2], b[3]])
        return results


# a=ObjDetRetinanet()
# img1=cv2.imread('/Users/pranoyr/PycharmProjects/eyes-age-alerts/detectors/object_detection/retinanet/images/people.jpg')
# img2=cv2.imread('/Users/pranoyr/PycharmProjects/eyes-age-alerts/detectors/object_detection/retinanet/images/coco2.png')

# img1=cv2.resize(img1,(640,480))
# img2=cv2.resize(img2,(640,480))

# imgs=[img1,img2]
# imgs=np.array(imgs)
# c=a.detect(imgs)
# print(c)

