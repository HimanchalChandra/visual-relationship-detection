import os
import json
from opts import parse_opts
# model = vgg16(pretrained=True)
# modules = list(model.children())[:-2]  
# base_net = nn.Sequential(*modules)

# print(base_net)
# import torch

opt = parse_opts()

	
with open(os.path.join(opt.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
	predicates = json.load(f)

int2word_pred = {}
for i, predicate in enumerate(predicates):
	int2word_pred[i] = predicate
print(int2word_pred)
