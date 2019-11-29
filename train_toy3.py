import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import argparse
import sys
import queue

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from knockknock import slack_sender


def main():
    NEG_SAMPLES = queue.Queue()
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=5, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=32, type=int)
    PARSER.add_argument("--train_modality_net", help="whether to train modality-specific network", default=0, type=int)
    PARSER.add_argument("--loss_function", help="which loss function", default=1, type=int)
    PARSER.add_argument("--verbose", help="print information", default=0, type=int)

    MY_ARGS = PARSER.parse_args()

    LOGGER.info("=============================================================")
    print(MY_ARGS)
    LOGGER.info("=============================================================")

    train_img = torch.load("cached_data/train_img")[:10]
    train_cap = torch.load("cached_data/train_cap")[:10]
    train_mask = torch.load("cached_data/train_mask")[:10]

    print("Loaded train data", train_img.size(), train_cap.size(), train_mask.size())

    NB_EPOCHS = MY_ARGS.epochs
    device = "cuda:0"

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 10),
        torch.nn.Softmax(),
    )
    ranking_loss = teacher_network.ContrastiveLoss(1.0, device)
    teacher_net.to(device)
    ranking_loss.to(device)

    # optimizer
    params_to_update_share = []

    for param in teacher_net.parameters():
        params_to_update_share.append(param)

    params_to_update = list(params_to_update_share)
    print(len(params_to_update))
    optimizer = optim.Adam(teacher_net.parameters(), lr=0.001)

    print("Start to train")
    for epoch in range(NB_EPOCHS):

        for idx in range(5):
            LOGGER.info("batch %d===========" % idx)
            samples = [idx, idx]
            img, cap, mask = tuple(t.to(device) for t in (train_img[samples],
                                                          train_cap[samples],
                                                          train_mask[samples]))

            if NEG_SAMPLES.empty():
                NEG_SAMPLES.put((img, cap, mask))
                continue
            else:
                _, neg_cap, neg_mask = NEG_SAMPLES.get()
                with torch.set_grad_enabled(False):
                    img_feature = vision_net.forward(img)
                    txt_feature = text_net.forward(cap, mask)
                    neg_txt_feature = text_net.forward(neg_cap, neg_mask)

                with torch.set_grad_enabled(True):
                    img_vec = teacher_net.forward(img_feature)
                    txt_vec = teacher_net.forward(txt_feature)
                    neg_sample = teacher_net.forward(neg_txt_feature)[0]

                    loss = ranking_loss(img_vec[0].view(1, 10),
                                        txt_vec[0].view(1, 10),
                                        neg_sample.view(1, 10))

                    LOGGER.error("Loss is %f" % loss.item())
                    loss.backward()
                    logits = ranking_loss.return_logits(img_vec[0].view(1, 10),
                                                        txt_vec[0].view(1, 10),
                                                        neg_sample.view(1, 10))
                    LOGGER.info("before============")
                    print(logits)

                    optimizer.step()
                    optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    img_vec = teacher_net.forward(img_feature)
                    txt_vec = teacher_net.forward(txt_feature)
                    logits = ranking_loss.return_logits(img_vec[0].view(1, 10),
                                                        txt_vec[0].view(1, 10),
                                                        neg_sample.view(1, 10))
                    LOGGER.info("after============")
                    print(logits)
            print()


if __name__ == '__main__':
    main()