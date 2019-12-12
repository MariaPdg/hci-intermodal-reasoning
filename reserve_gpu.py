import torch


u1 = torch.rand((5000*5000, 40), device="cuda:0")
print(u1.size())
while True:
    pass