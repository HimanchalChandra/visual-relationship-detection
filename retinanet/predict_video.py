# import keras
import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots', 'face_det_model.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
names_to_labels = {
    'face':0
    # 'smoking'   : 0,
    # 'writing_on_a_book' :1,
    # 'phoning':2,
    # 'drinking':3,
    # 'waving_hands':4,
    # 'jumping':5,
    # 'riding_a_bike':6,
    # 'walking':7,
    # 'climbing':8,
    # 'reading':9,
    # 'texting_message':10,
    # 'taking_photos':11,
    # 'applauding':12,
    # 'playing_guitar':13,
    # 'shooting_an_arrow':14,
    # 'feeding_a_horse':15,
    # 'looking_through_a_telescope':16,
    # 'cutting_vegetables':17,
    # 'running':18,
    # 'rowing_a_boat':19,
    # 'cutting_trees':20,
    # 'looking_through_a_microscope':21,
    # 'riding_a_horse':22,
    # 'watching_TV':23,
    # 'cooking':24,
    # 'washing_dishes':25,
    # 'holding_an_umbrella':26,
    # 'walking_the_dog':27,
    # 'gardening':28,
    # 'using_a_computer':29,
    # 'cleaning_the_floor':30,
    # 'pushing_a_cart':31,
    # 'fixing_a_car':32,
    # 'throwing_frisby':33,
    # 'brushing_teeth':34,
    # 'pouring_liquid':35,
    # 'fixing_a_bike':36,
    # 'blowing_bubbles':37,
    # 'playing_violin':38,
    # 'fishing':39,
    # 'writing_on_a_board':40
    }

labels_to_names = {v: k for k, v in names_to_labels.items()}


camera=cv2.VideoCapture('/Users/pranoyr/Downloads/DFMD 1 .asf')
while True:
    ret,image=camera.read()
    print(image.shape)
    # load image
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)


    print(boxes.shape)
    print(scores.shape)
    print(labels.shape)
    # correct for image scale
    boxes /= scale
    c=0
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.7:
            break
        c+=1
        color = label_color(label)

        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    print(c)  

    #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imshow('window',draw)
    cv2.waitKey(1)
