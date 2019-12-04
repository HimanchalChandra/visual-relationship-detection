import torch
from torch.nn import MarginRankingLoss

criterion = MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')


x1 = torch.Tensor(32,3)
print(x1)
x2 = torch.Tensor(32,3)
y = torch.ones([32, 3])

loss = criterion(x1,x2,y)
print(loss)