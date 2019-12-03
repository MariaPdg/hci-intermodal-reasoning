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
teacher_net1 = teacher_network.TeacherNet()
teacher_net2 = teacher_network.TeacherNet()
ranking_loss = teacher_network.ContrastiveLoss(DELTA, device)
teacher_net1.to(device)
teacher_net2.to(device)


teacher_net1.load_state_dict(torch.load("models/enc1-8296-norm"))
teacher_net2.load_state_dict(torch.load("models/enc2-8296-norm"))


print("Start to evaluate")
for nb_neg_samples in [8, 16, 32, 64, 128, 256]:
    running_loss = 0.0
    start_time = time.time()
    running_corrects = 0.0
    total_samples = val_img.size(0)
    for i in range(val_img.size(0)):
        samples = [i, i]
        predictions = []

        while len(samples) < nb_neg_samples+2:
            neg_idx = random.choice(range(val_img.size(0)))
            if neg_idx != i:
                samples.append(neg_idx)
            else:
                continue

        img, cap, mask = tuple(t.to(device) for t in (val_img[samples],
                                                      val_cap[samples],
                                                      val_mask[samples]))
        # with torch.set_grad_enabled(False):
        #     img_vec = teacher_net1.forward(vision_net.forward(img[:2]))
        #     txt_vec = teacher_net2.forward(text_net.forward(cap[:2], mask[:2]))
        #     neg_txt_vec = teacher_net2.forward(text_net.forward(cap[2:], mask[2:]))
        #     neg_txt_vec = torch.transpose(neg_txt_vec, 0, 1)
        #     print(img_vec)
        #     print("1", ranking_loss.return_logits(img_vec, txt_vec, neg_txt_vec))
        #
        # with torch.set_grad_enabled(False):
        #     img_vec = teacher_net1.forward(vision_net.forward(img[1:]))
        #     txt_vec = teacher_net2.forward(text_net.forward(cap[1:], mask[1:]))
        #     mul = torch.bmm(img_vec.view(img_vec.size(0), 1, img_vec.size(1)),
        #                     txt_vec.view(img_vec.size(0), img_vec.size(1), 1))
        #     mul = mul.view(mul.size(0), 1)
        #     print(img_vec[0])
        #     print("2", mul)

        logits = []
        with torch.set_grad_enabled(False):
            img_vec = teacher_net1.forward(vision_net.forward(img))
            txt_vec = teacher_net2.forward(text_net.forward(cap, mask))
            key = img_vec[0]
            for du in range(1, img_vec.size(0)):
                logits.append(torch.matmul(key, txt_vec[du]).item())

        pred = np.argmax(logits)
        if pred == 0:
            running_corrects += 1


    LOGGER.info("Val acc = %f (%d/%d) attacking vectors = %d" % (float(running_corrects / total_samples), running_corrects, total_samples, nb_neg_samples))
    LOGGER.info("Evaluation done in %f mins" % ((time.time() - start_time) / 60))
