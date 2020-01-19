import csv
from sklearn.metrics import recall_score
import torch
import numpy as np
from sklearn.metrics import average_precision_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


class Metric:
    """ Computer precision/recall for multilabel classifcation
    """

    def __init__(self, num_classes):
        # For each class
        self.precision = dict()
        self.recall = dict()
        self.average_precision = dict()
        self.gt = []
        self.y = []
        self.num_classes = num_classes
    
    def update(self, outputs, targets):
        self.y.append(outputs.detach().cpu())
        self.gt.append(targets.detach().cpu())

    def compute_metrics(self):
        preds = torch.cat(self.y)
        targets = torch.cat(self.gt)
        preds = preds.numpy()
        targets = targets.numpy()

        #preds = np.argmax(preds, axis=1)
        # # one hot encode
        # onehot_encoded = list()
        # for value in preds:
        #     letter = [0 for _ in range(self.num_classes)]
        #     letter[value] = 1
        #     onehot_encoded.append(letter)
        # preds = np.array(onehot_encoded)

        onehot_encoded = list()
        for value in targets:
            letter = [0 for _ in range(self.num_classes)]
            letter[value] = 1
            onehot_encoded.append(letter)
        targets = np.array(onehot_encoded)

        # recall = recall_score(targets, preds, average='micro')
        self.average_precision["micro"] = average_precision_score(targets, preds,
                                                                  average="micro")
        
        recall = self.average_precision
        for i in range(self.num_classes):
            self.average_precision[i] = average_precision_score(targets[:, i], preds[:, i])

        print(self.average_precision)
        
        return recall