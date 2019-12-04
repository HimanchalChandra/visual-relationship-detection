# Visual Relationship Detection

## Getting Started
### Prerequisites
1. Pytorch

### Download the pre-trained weights

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

## Reference
https://arxiv.org/abs/1904.07798


## License
This project is licensed under the MIT License 


