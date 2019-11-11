import torchvision.transforms as transforms
import torchvision.datasets as datasets
import types
import torch
import utils
import text_network
import teacher_network
import vision_network

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


train_img = torch.load("cached_data/train_img")
train_cap = torch.load("cached_data/train_cap")
train_mask = torch.load("cached_data/train_mask")

val_img = torch.load("cached_data/val_img")
val_cap = torch.load("cached_data/val_cap")
val_mask = torch.load("cached_data/val_mask")


BATCH_SIZE = 3
train_data = TensorDataset(train_img, train_cap, train_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
valid_data = TensorDataset(val_img, val_cap, val_mask)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE * 8, num_workers=2)

device = "cpu"
text_net = text_network.TextNet(device)
vision_net = vision_network.VisionNet(device)
teacher_net = teacher_network.TeacherNet()
teacher_net.to(device)

for step, batch in enumerate(train_dataloader):
    img, cap, mask = tuple(t.to(device) for t in batch)
    img_vec = teacher_net.forward(vision_net.forward(img))
    txt_vec = teacher_net.forward(text_net.forward(cap, mask))

    break

