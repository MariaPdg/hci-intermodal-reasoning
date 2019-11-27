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
teacher_net = teacher_network.TeacherNet()
ranking_loss = teacher_network.ContrastiveLoss(DELTA, device)
teacher_net.to(device)

# optimizer
params_to_update_share = []
params_to_update_img = vision_net.parameters()
params_to_update_txt = []


params_to_update = list(params_to_update_share) + list(params_to_update_img) + list(params_to_update_txt)
optimizer = optim.Adam(params_to_update, lr=0.0001)

teacher_net.load_state_dict(torch.load("models/156-train_modality0"))

start_time = time.time()
print("Start to evaluate")
for _ in range(2):
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0

    for step, batch in enumerate(valid_dataloader):
        img, cap, mask = tuple(t.to(device) for t in batch)
    
        with torch.set_grad_enabled(False):
    
            img_vec = teacher_net.forward(vision_net.forward(img))
            txt_vec = teacher_net.forward(text_net.forward(cap, mask))
    
            preds = ranking_loss.predict(img_vec, txt_vec)

        print(preds)
        running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
        total_samples += len(preds)
        break
        
    LOGGER.info("Val acc = %f (%d/%d)" % (float(running_corrects/total_samples), running_corrects, total_samples))
LOGGER.info("Evaluation done in %f mins" % ((time.time()-start_time)/60))
