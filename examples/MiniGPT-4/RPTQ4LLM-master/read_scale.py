import torch

for i in range(32):
    layer = torch.load(f"output/qlayer_{i}.pth")

    print("-----------------------------")
    for name, value in layer.items():
        print(name, value.shape)