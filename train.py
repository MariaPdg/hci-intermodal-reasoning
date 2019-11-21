import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import argparse
import sys

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from knockknock import slack_sender

"""
for sending notification when your code finishes
"""
sys.stdin = open("webhook_url.txt", "r")
SLACK_WEBHOOK = sys.stdin.readline().rstrip()


@slack_sender(webhook_url=SLACK_WEBHOOK, channel="bot")
def main():
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=1, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=4, type=int)
    PARSER.add_argument("--train_modality_net", help="whether to train modality-specific network", default=0, type=int)
    PARSER.add_argument("--loss_function", help="which loss function", default=0, type=int)
    PARSER.add_argument("--verbose", help="print information", default=0, type=int)
    
    MY_ARGS = PARSER.parse_args()
    
    LOGGER.info("=============================================================")
    print(MY_ARGS)
    LOGGER.info("=============================================================")

    
    train_img = torch.load("cached_data/train_img")
    train_cap = torch.load("cached_data/train_cap")
    train_mask = torch.load("cached_data/train_mask")
    
    val_img = torch.load("cached_data/val_img")
    val_cap = torch.load("cached_data/val_cap")
    val_mask = torch.load("cached_data/val_mask")
    
    print("Loaded train data", train_img.size(), train_cap.size(), train_mask.size())
    print("Loaded val data", val_img.size(), val_cap.size(), val_mask.size())
    
    DELTA = 0.07
    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    train_modality_net = bool(MY_ARGS.train_modality_net)
    device = "cuda:1"
    verbose = bool(MY_ARGS.verbose)
    LOSS_FUNCTIONS = {0: teacher_network.RankingLossFunc(DELTA), 1: teacher_network.ContrastiveLoss(DELTA, device)}
    
    train_data = TensorDataset(train_img, train_cap, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE * 2, num_workers=2)
    
    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net = teacher_network.TeacherNet()
    ranking_loss = LOSS_FUNCTIONS[MY_ARGS.loss_function]
    teacher_net.to(device)
    ranking_loss.to(device)
    
    # optimizer
    params_to_update_share = []
    params_to_update_img = vision_net.parameters()
    params_to_update_txt = []
    
    params_to_update = list(params_to_update_share) + list(params_to_update_img) + list(params_to_update_txt)
    optimizer = optim.Adam(params_to_update, lr=0.0001)
    
    print("Start to train")
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(NB_EPOCHS):
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
    
        for step, batch in enumerate(train_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)
    
            with torch.set_grad_enabled(train_modality_net):
                img_feature = vision_net.forward(img)
                txt_feature = text_net.forward(cap, mask)
    
            with torch.set_grad_enabled(True):
                img_vec = teacher_net.forward(img_feature)
                txt_vec = teacher_net.forward(txt_feature)
    
                loss = ranking_loss(img_vec, txt_vec)
                preds = ranking_loss.predict(img_vec, txt_vec)
                try:
                    loss.backward()
                except AttributeError:
                    img, cap, mask, img_vec, txt_vec = tuple(t.to("cpu") for t in (img, cap, mask, img_vec, txt_vec))
                    print("Faulty", img.size(), cap.size(), mask.size(), img_vec.size(), txt_vec.size())
                    torch.save(img, "img_faulty")
                    torch.save(cap, "cap_faulty")
                    torch.save(mask, "mask_faulty")
                    torch.save(img_vec, "img_vec_faulty")
                    torch.save(txt_vec, "txt_vec_faulty")
                    continue
                optimizer.step()
                optimizer.zero_grad()
    
            running_loss += loss.item() * BATCH_SIZE
            running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)
        
        if verbose:
            LOGGER.info("Epoch %d: train loss = %f" % (epoch, running_loss))
            LOGGER.info(
                "          train acc = %f (%d/%d)" % (float(running_corrects / total_samples), running_corrects, total_samples))
    
        train_losses.append(running_loss)
        train_accs.append(float(running_corrects / total_samples))
    
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
    
        for step, batch in enumerate(valid_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)
    
            with torch.set_grad_enabled(False):
                img_vec = teacher_net.forward(vision_net.forward(img))
                txt_vec = teacher_net.forward(text_net.forward(cap, mask))
    
                loss = ranking_loss(img_vec, txt_vec)
                preds = ranking_loss.predict(img_vec, txt_vec)
    
            running_loss += loss.item() * BATCH_SIZE
            running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)
        
        if verbose or epoch == NB_EPOCHS-1:
            LOGGER.info("Val loss = %f" % running_loss)
            LOGGER.info("Val acc = %f (%d/%d)" % (float(running_corrects / total_samples), running_corrects, total_samples))
        val_losses.append(running_loss)
        val_accs.append(float(running_corrects / total_samples))
    
    LOGGER.info("Training done in %f mins" % ((time.time() - start_time) / 60))
    
    model_name = "%.2f-train_modality%d" % (float(running_corrects / total_samples), train_modality_net)
    torch.save(teacher_net.state_dict(), "models/%s" % model_name)
    print(train_losses)
    print(train_accs)
    print(val_losses)
    print(val_accs)
    print()
    return val_accs[-1], MY_ARGS


if __name__ == '__main__':
    main()
