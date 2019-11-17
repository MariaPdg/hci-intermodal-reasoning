import torchvision.transforms as transforms
import torchvision.datasets as datasets
import types
import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import sys


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


train_img = torch.load("img_faulty", map_location={'cuda:0': 'cpu'})
train_cap = torch.load("cap_faulty", map_location={'cuda:0': 'cpu'})
train_mask = torch.load("mask_faulty", map_location={'cuda:0': 'cpu'})

val_img = torch.load("cached_data/val_img")
val_cap = torch.load("cached_data/val_cap")
val_mask = torch.load("cached_data/val_mask")

print("Loaded train data", train_img.size(), train_cap.size(), train_mask.size())
print("Loaded val data", val_img.size(), val_cap.size(), val_mask.size())

DELTA = 0.002
BATCH_SIZE = 8
NB_EPOCHS = 1
LOGGER = utils.Logger()

train_data = TensorDataset(train_img, train_cap, train_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
valid_data = TensorDataset(val_img, val_cap, val_mask)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE * 2, num_workers=2)

device = "cpu"
text_net = text_network.TextNet(device)
vision_net = vision_network.VisionNet(device)
teacher_net = teacher_network.TeacherNet()
ranking_loss = teacher_network.RankingLossFunc(DELTA)
teacher_net.to(device)
ranking_loss.to(device)

# optimizer
params_to_update_share = []
params_to_update_img = vision_net.parameters()
params_to_update_txt = []

for name, param in teacher_net.named_parameters():
    if param.requires_grad is True:
        params_to_update_share.append(param)
        print("\t", name)


for name, param in vision_net.named_parameters():
    if param.requires_grad is True:
        print("\t", name)

for name, param in text_net.named_parameters():
    if param.requires_grad is True:
        params_to_update_txt.append(param)
        print("\t", name)


params_to_update = list(params_to_update_share) + list(params_to_update_img) + list(params_to_update_txt)
optimizer = optim.Adam(params_to_update, lr=0.0001)

print("Start to train")
start_time = time.time()
for epoch in range(NB_EPOCHS):
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0

    for step, batch in enumerate(train_dataloader):
        img, cap, mask = tuple(t.to(device) for t in batch)
    
        with torch.set_grad_enabled(True):
    
            img_vec = teacher_net.forward(vision_net.forward(img))
            txt_vec = teacher_net.forward(text_net.forward(cap, mask))
    
            loss = ranking_loss(img_vec, txt_vec)
            preds = teacher_net.predict(img_vec, txt_vec)
            try:
                loss.backward()
            except AttributeError:
                img, cap, mask, img_vec, txt_vec = tuple(t.to("cpu") for t in (img, cap, mask, img_vec, txt_vec))
                print("Faulty", img.size(), cap.size(), mask.size(), img_vec.size(), txt_vec.size())
                sys.exit()
            optimizer.step()
            optimizer.zero_grad()
    
    
        running_loss += loss.item() * BATCH_SIZE
        running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
        total_samples += len(preds)
        
    LOGGER.info("Epoch %d: train loss = %f" % (epoch, running_loss))
    LOGGER.info("          train acc = %f (%d/%d)" % (float(running_corrects/total_samples), running_corrects, total_samples))

LOGGER.info("Training done in %f mins" % ((time.time()-start_time)/60))