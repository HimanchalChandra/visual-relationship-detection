# On Exploring Undetermined Relationships for Visual Relationship Detection

## Getting Started
### Prerequisites
1. Pytorch
2. Keras RetinaNet https://github.com/fizyr/keras-retinanet

### Download the pre-trained weights
https://drive.google.com/file/d/1tZJvOsta6CPIwz5D1ruZGIAncab0Nx2O/view?usp=sharing

### Download the pre-trained weights for Retina Net
https://www.dropbox.com/s/hdwd6qjir032ktj/resnet50_vrd_inference.h5?dl=0

### Download Glove word vectors
https://nlp.stanford.edu/projects/glove/

### Dataset
1) VRD - Download from  https://cs.stanford.edu/people/ranjaykrishna/vrd/
2) Visual Genome (To do)

 
### Training
```
python main.py   --epochs 1000 --gpu 0 --save_interval 5 --dataset_path ./data/vrd --glove_path ./data/glove.6B/glove.6B.300d.txt
```

### Inference
```
python predict.py --dataset_path ./data/vrd --glove_path ./data/glove.6B/glove.6B.300d.txt


```

### Results
![alt text](./outputs/sf.jpg)


1) wheel next to person ,score:0.7031404376029968
2) wheel below car ,score:0.9070525765419006
3) person wear helmet ,score:0.973259449005127
4) person above wheel ,score:0.798913836479187
5) person wear jacket ,score:0.8361937403678894
6) person wear helmet ,score:0.9548267126083374
7) person near car ,score:0.8904088735580444
8) person hold person ,score:0.9253084063529968
9) person on bike ,score:0.9804078735580444


TO DO
give annotations if detections are empty 


## Reference
Visual Relationship Detection with Language prior and Softmax
https://arxiv.org/abs/1904.07798


## License
This project is licensed under the Apache License 2.0


