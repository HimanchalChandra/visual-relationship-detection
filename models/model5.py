import torch
import torch.nn as nn
from torchvision import models
import json
import numpy as np
import torch.nn.functional as F
from opts import parse_opts
import os

opt = parse_opts()

with open(os.path.join(opt.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
    objects = json.load(f)

word2int = {}
for i, obj in enumerate(objects):
    word2int[obj] = i

glove = {}
vocab = len(objects)
matrix_len = vocab
weights_matrix = np.zeros((matrix_len, 300))
with open(opt.glove_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        glove[word] = vect
for i, obj in enumerate(objects):
    try:
        weights_matrix[word2int[obj]] = glove[obj]
    except KeyError:
        weights_matrix[word2int[obj]] = np.random.normal(
            scale=0.6, size=(300, ))
weights_matrix = torch.Tensor(weights_matrix)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, embedding_dim


class LanguageModule(nn.Module):
    """ Language Moddule"""

    def __init__(self, num_classes):
        super(LanguageModule, self).__init__()
        self.fc1 = nn.Linear(611, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x

# class LanguageModule(nn.Module):
#     """ Language Moddule"""

#     def __init__(self, hidden_dim, target_size):
#         super(LanguageModule, self).__init__()
#         #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.word_embeddings, embedding_dim = create_emb_layer(
#             weights_matrix, True)
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,
#                             num_layers=1, batch_first=True)
#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim, target_size)

#     def forward(self, x):
#         x = self.word_embeddings(x)
#         lstm_out, _ = self.lstm(x)
#         x = lstm_out[:, -1, :]
#         x = self.hidden2tag(x)
#         #x = F.log_softmax(x, dim=1)
#         return x


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]
        self.vm = nn.Sequential(*modules)

        modules = [nn.Linear(model.fc.in_features + 11, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True)]
        modules += [nn.Linear(4096, num_classes)]
        self.vm_sp = nn.Sequential(*modules)

        self.word_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, True)
        self.lm = LanguageModule(num_classes=num_classes)


    def forward(self, img, spatial_locations, word_vectors):
        vm_out = self.vm(img)
        vm_out = vm_out.view(vm_out.size(0), -1)
        vm_sp_features = torch.cat([vm_out, spatial_locations], dim=1)
        vm_sp_out = self.vm_sp(vm_sp_features)


        # compute word embedding
        word_vectors = self.word_embeddings(word_vectors)
        # compute input features for language module
        word_vectors = word_vectors.view(word_vectors.size(0), -1)
        word_sp_vector = torch.cat([word_vectors, spatial_locations], dim=1)
        lm_out = self.lm(word_sp_vector)
    
        x = torch.mul(vm_sp_out, lm_out)
        return x


if __name__ == "__main__":
    net = Net(num_classes=100)
    print(net.parameters)