import os
import json
from torchvision.models import vgg16
import torch.nn as nn
import torch

model = vgg16(pretrained=False)
modules = list(model.children())[:-1]  
base_net = nn.Sequential(*modules)


print(base_net)

# x = torch.Tensor(1,3,224,224)

# x = base_net(x)
# print(x.shape)



# opt = parse_opts()

	
# with open(os.path.join(opt.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
# 	predicates = json.load(f)

# int2word_pred = {}
# for i, predicate in enumerate(predicates):
# 	int2word_pred[i] = predicate
# print(int2word_pred)
