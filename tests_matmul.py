import torch
import time

a = torch.rand((100, 1), device="cpu").squeeze()
print(torch.argsort(a))
print(a[torch.argsort(a)])