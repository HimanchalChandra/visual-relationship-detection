import torch
checkpoint = torch.load('./snapshots/model_acc_91.pth')
model = checkpoint['model_state_dict']
new_model = model.copy()
for k, v in model.items():
    if k in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']:
        new_model.pop(k)

state = {'epoch': 0, 'model_state_dict': new_model, 'optimizer_state_dict': {}}
torch.save(state, './snapshots/model.pth')

