import torch
import torch.nn as nn
from torchvision import models
import json
import numpy as np
import torch.nn.functional as F


glove_path = '../glove.6B'
with open('../visual_genome/json_dataset/objects.json', 'r') as f:
    objects = json.load(f)

word2int = {}
for i, obj in enumerate(objects):
    word2int[obj] = i

vocab = len(objects)
matrix_len = vocab
weights_matrix = np.zeros((matrix_len, 300))

glove = {}
with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        glove[word] = vect

for i, obj in enumerate(objects):
    try:
        weights_matrix[word2int[obj]] = glove[obj]
    except KeyError:
        weights_matrix[word2int[obj]] = np.random.normal(scale=0.6, size=(300, ))

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=5)

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
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        vm = models.resnet50(pretrained=True)
        self.vm = torch.nn.Sequential(*(list(vm.children())[:-2]))
        self.lm = LanguageModule(300, 512)

    
        self.fc1 = nn.Linear(100872, 4096)  
        self.fc2 = nn.Linear(4096, 4096) 
        self.fc3 = nn.Linear(4096, num_classes)


    def forward(self, img, spatial_locations, word_vectors):
        features_img = self.vm(img)
        features_wordvec = self.lm(word_vectors)

        features_img = features_img.view(features_img.size(0), -1)
        features_wordvec = features_wordvec.view(features_wordvec.size(0), -1)
        spatial_locations = spatial_locations.view(spatial_locations.size(0), -1)

        combined_features = torch.cat([features_img, features_wordvec, spatial_locations], dim=1)

        x = self.fc1(combined_features)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x 

if __name__ == "__main__":
    net = Net(num_classes=100)
    print(net.parameters)