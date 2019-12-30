import torch
import torch.nn.functional
import utils
import text_network
import teacher_network
import vision_network
import time
import random
import numpy as np
import sys

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

val_img = torch.load("cached_data/train_img")
val_cap = torch.load("cached_data/train_cap")
val_mask = torch.load("cached_data/train_mask")

print("Loaded val data", val_img.size(), val_cap.size(), val_mask.size())

BATCH_SIZE = 128
LOGGER = utils.Logger()

valid_data = TensorDataset(val_img, val_cap, val_mask)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE * 2, num_workers=2)

device = "cuda:0"
text_net = text_network.TextNet(device)
vision_net = vision_network.VisionNet(device)

teacher_net1 = teacher_network.TeacherNet()
teacher_net2 = teacher_network.TeacherNet()
teacher_net1.to(device)
teacher_net2.to(device)


teacher_net1.load_state_dict(torch.load("models/enc1-8296-norm"))
teacher_net2.load_state_dict(torch.load("models/enc2-8296-norm"))
vision_net.model.load_state_dict(torch.load("models/enc1-8296-norm"))
text_net.model.load_state_dict(torch.load("models/enc2-8296-norm"))

text_net.model.eval()
vision_net.model.eval()
teacher_net1.eval()
teacher_net2.eval()

print("Start to evaluate")
img_vecs = []
txt_vecs = []
with torch.no_grad():
    for step, batch in enumerate(valid_dataloader):
        img, cap, mask = tuple(t.to(device) for t in batch)
        img_vec = teacher_net1.forward(vision_net.forward(img))
        txt_vec = teacher_net2.forward(text_net.forward(cap, mask))

        img_vecs.append(img_vec)
        txt_vecs.append(txt_vec)

img_vecs = torch.cat(img_vecs, dim=0)
txt_vecs = torch.cat(txt_vecs, dim=0)
