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

teacher_net1 = teacher_network.TeacherNet()
teacher_net2 = teacher_network.TeacherNet()
ranking_loss = teacher_network.ContrastiveLoss(DELTA, device)
teacher_net1.to(device)
teacher_net2.to(device)


teacher_net1.load_state_dict(torch.load("models/enc1-8296-norm"))
teacher_net2.load_state_dict(torch.load("models/enc2-8296-norm"))

vision_net.model.eval()
text_net.model.eval()
teacher_net1.eval()
teacher_net2.eval()

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
    print(img_vec.size(), txt_vec.size(), neg_txt_vec.size())
    print("1", ranking_loss.return_logits(img_vec, txt_vec, neg_txt_vec))

with torch.set_grad_enabled(False):
    img_vec = teacher_net1.forward(vision_net.forward(img[:2]))
    txt_vec = teacher_net2.forward(text_net.forward(cap[1:], mask[1:]))
    print(img_vec.size(), txt_vec.size())
    txt_vec = torch.transpose(txt_vec, 0, 1)
    mul = torch.matmul(img_vec, txt_vec)
    print("2", mul)

logits = []
img_vec = teacher_net1.forward(vision_net.forward(img))
txt_vec = teacher_net2.forward(text_net.forward(cap, mask))
key = img_vec[0]
for du in range(1, img_vec.size(0)):
    logits.append(torch.matmul(key, txt_vec[du]).item())
pred = np.argmax(logits)
print(pred, pred == 0)
print("3", logits)
