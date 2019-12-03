import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import argparse
import numpy as np
import sys
import queue as queue_lib

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from knockknock import slack_sender


def main():
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=100, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=32, type=int)
    PARSER.add_argument("--train_modality_net", help="whether to train modality-specific network", default=0, type=int)
    PARSER.add_argument("--loss_function", help="which loss function", default=1, type=int)
    PARSER.add_argument("--verbose", help="print information", default=1, type=int)

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

    MOMENT = 0.9
    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    device = "cuda:0"

    train_data = TensorDataset(train_img, train_cap, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=2)

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net1 = teacher_network.TeacherNet()
    teacher_net2 = teacher_network.TeacherNet()
    ranking_loss = teacher_network.ContrastiveLoss(1, device)
    teacher_net1.to(device)
    teacher_net2.to(device)
    ranking_loss.to(device)

    param_names = teacher_net1.state_dict().keys()

    # optimizer
    optimizer = optim.Adam(teacher_net1.parameters(), lr=3e-5)

    print("Start to train")
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    NEG_SAMPLES = teacher_network.CustomedQueue()
    QUEUE_SIZE = 64

    for epoch in range(NB_EPOCHS):
        """
        Training
        """
        running_loss = []
        running_corrects = 0.0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)
            if NEG_SAMPLES.empty():
                with torch.set_grad_enabled(False):
                    txt_vec = teacher_net2.forward(text_net.forward(cap, mask))
                    NEG_SAMPLES.enqueue(txt_vec)
                continue

            else:
                with torch.set_grad_enabled(False):
                    img_feature = vision_net.forward(img)
                    txt_feature = text_net.forward(cap, mask)

                with torch.set_grad_enabled(True):
                    img_vec = teacher_net1.forward(img_feature)
                    txt_vec = teacher_net2.forward(txt_feature)
                    neg_txt_vec = NEG_SAMPLES.get_tensor()
                    neg_txt_vec = neg_txt_vec.detach()
                    txt_vec = txt_vec.detach()

                    # print("img", img_vec)
                    loss = ranking_loss(img_vec, txt_vec, neg_txt_vec)
                    running_loss.append(loss.item())
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(parameters=teacher_net1.parameters(),
                                                   max_norm=1.0)
                    # for param in teacher_net1.parameters():
                    #     print(param.grad.data.sum())

                    optimizer.step()
                    optimizer.zero_grad()
                    # print()

                with torch.set_grad_enabled(False):
                    img_vec = teacher_net1.forward(img_feature)
                    txt_vec = teacher_net2.forward(txt_feature)
                    preds = ranking_loss.predict(img_vec, txt_vec)
                    NEG_SAMPLES.enqueue(txt_vec)

                for key in param_names:
                    teacher_net2.state_dict()[key] = teacher_net2.state_dict()[key] * MOMENT + \
                                                     (1 - MOMENT) * teacher_net1.state_dict()[key]

                running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
                total_samples += len(preds)

            if NEG_SAMPLES.size >= QUEUE_SIZE:
                NEG_SAMPLES.dequeue(BATCH_SIZE)

        LOGGER.info("Epoch %d: train loss = %f, max=%f min=%f" % (epoch, np.average(running_loss),
                                                                  np.max(running_loss),
                                                                  np.min(running_loss)))
        LOGGER.info(
            "          train acc = %f (%d/%d)" % (
                float(running_corrects / total_samples), running_corrects, total_samples))

        train_losses.append(np.average(running_loss))
        train_accs.append(float(running_corrects / total_samples))

    model_name = "%d" % running_corrects
    torch.save(teacher_net1.state_dict(), "models/enc1-%s-norm" % model_name)
    torch.save(teacher_net2.state_dict(), "models/enc2-%s-norm" % model_name)

    print(train_losses)
    print(train_accs)
    print()


if __name__ == '__main__':
    main()
