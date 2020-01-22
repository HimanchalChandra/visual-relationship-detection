import torch

a = torch.Tensor(2,3,4)
b = torch.Tensor(2,3,)

a.size()  # 2, 3, 4
b.size()  # 2, 3
# b = torch.unsqueeze(b, dim=2)  # 2, 3, 1
# torch.unsqueeze(b, dim=-1) does the same thing

print(a.shape)
print(b.shape)

c = torch.stack([a, b], dim=2)  # 2, 3, 5



print(c.shape)