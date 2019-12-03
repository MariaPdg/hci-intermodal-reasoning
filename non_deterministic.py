import torch
import utils
import text_network
import teacher_network
import vision_network
import time
import random
import numpy as np
import sys

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

val_img = torch.load("cached_data/val_img")
val_cap = torch.load("cached_data/val_cap")
val_mask = torch.load("cached_data/val_mask")

print("Loaded val data", val_img.size(), val_cap.size(), val_mask.size())

DELTA = 0.002
BATCH_SIZE = 8
NB_EPOCHS = 1
LOGGER = utils.Logger()

valid_data = TensorDataset(val_img, val_cap, val_mask)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE * 2, num_workers=2)

device = "cuda:0"
text_net = text_network.TextNet(device)
vision_net = vision_network.VisionNet(device)
vision_net.model.eval()
teacher_net1 = teacher_network.TeacherNet()
teacher_net2 = teacher_network.TeacherNet()
ranking_loss = teacher_network.ContrastiveLoss(DELTA, device)
teacher_net1.to(device)
teacher_net2.to(device)


teacher_net1.load_state_dict(torch.load("models/enc1-8296-norm"))
teacher_net2.load_state_dict(torch.load("models/enc2-8296-norm"))


print("Start to evaluate")
samples = [0, 0]


predictions = []
nb_neg_samples = 3
while len(samples) < nb_neg_samples+2:
    neg_idx = random.choice(range(val_img.size(0)))
    if neg_idx != 0:
        samples.append(neg_idx)
    else:
        continue

img, cap, mask = tuple(t.to(device) for t in (val_img[samples],
                                              val_cap[samples],
                                              val_mask[samples]))

with torch.set_grad_enabled(False):
    img_vec = teacher_net1(vision_net.forward(img[:2]))
    txt_vec = teacher_net2(text_net.forward(cap[:2], mask[:2]))
    neg_txt_vec = teacher_net2(text_net.forward(cap[2:], mask[2:]))
    neg_txt_vec = torch.transpose(neg_txt_vec, 0, 1)
    print(img_vec[0])
    # print("1", ranking_loss.return_logits(img_vec, txt_vec, neg_txt_vec))

with torch.set_grad_enabled(False):
    img_vec = teacher_net1.forward(vision_net.forward(img[1:]))
    txt_vec = teacher_net2.forward(text_net.forward(cap[1:], mask[1:]))
    # mul = torch.bmm(img_vec.view(img_vec.size(0), 1, img_vec.size(1)),
    #                 txt_vec.view(img_vec.size(0), img_vec.size(1), 1))
    # mul = mul.view(mul.size(0), 1)
    print(img_vec[0])
    # print("2", mul)

logits = []
with torch.set_grad_enabled(False):
    img_vec = teacher_net1.forward(vision_net.forward(img))
    txt_vec = teacher_net2.forward(text_net.forward(cap, mask))
    key = img_vec[0]
    for du in range(1, img_vec.size(0)):
        logits.append(torch.matmul(key, txt_vec[du]).item())

v1 = torch.rand((1, 2048), device="cuda:0")
v2 = torch.rand((1, 2048), device="cuda:0")
v3 = torch.rand((1, 2048), device="cuda:0")
b1 = torch.cat([v1, v2], dim=0)
b2 = torch.cat([v1, v2, v3], dim=0)
print(b1.size(), b2.size())

with torch.set_grad_enabled(False):
    r1 = teacher_net1(b1)
    r2 = teacher_net1(b2)
    print(r1[0])
    print(r2[0])

# for i in range(3):
#     with torch.set_grad_enabled(False):
#
#         img_vec = vision_net.forward(img)
#         txt_vec = text_net.forward(cap, mask)
#         vec1 = teacher_net1(img_vec)
#         vec2 = teacher_net2(txt_vec)
#         vec3 = teacher_net1(vision_net.forward(img))
#         print(img_vec[0])
#         print(txt_vec[0])
#         print(vec1[0])
#         print(vec2[0])
#         print(vec3[0])
#         print()