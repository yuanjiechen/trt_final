import torch

weight = torch.randn([20, 2])
torch.save(weight, "./weight.pth")