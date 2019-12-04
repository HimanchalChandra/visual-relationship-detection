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

    def __init__(self, hidden_dim, target_size):
        super(LanguageModule, self).__init__()
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, True)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, x):
        x = self.word_embeddings(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.hidden2tag(x)
        #x = F.log_softmax(x, dim=1)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        vm = models.vgg16(pretrained=True)
        self.vm = torch.nn.Sequential(*(list(vm.children())[:-1]))
        self.lm = LanguageModule(300, 512)
        self.fc1 = nn.Linear(25093, 4096)
        self.fc1_bn = nn.BatchNorm1d(4096)
        self.fc1_final = nn.Linear(4096, num_classes)
        self.fc2 = nn.Linear(517, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)
        self.fc2_final = nn.Linear(4096, num_classes)

    def forward(self, img, spatial_locations, word_vectors):
        vm_out = self.vm(img)
        lm_out = self.lm(word_vectors)
        vm_out = vm_out.view(vm_out.size(0), -1)
        lm_out = lm_out.view(lm_out.size(0), -1)

        spatial_locations = spatial_locations.view(
            spatial_locations.size(0), -1)
        vm_sp_combined = torch.cat([vm_out, spatial_locations], dim=1)
        lm_sp_combined = torch.cat([lm_out, spatial_locations], dim=1)

        x1 = self.fc1(vm_sp_combined)
        x1 = self.fc1_bn(x1)
        x1 = F.relu(x1)
        x1 = self.fc1_final(x1)

        x2 = self.fc2(lm_sp_combined)
        x2 = self.fc2_bn(x2)
        x2 = F.relu(x2)
        x2 = self.fc2_final(x2)

        x = torch.mul(x1, x2)
        return x


if __name__ == "__main__":
    net = Net(num_classes=100)
    print(net.parameters)
