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


def main():
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

    train_img = torch.load("cached_data/train_img")
    train_cap = torch.load("cached_data/train_cap")
    train_mask = torch.load("cached_data/train_mask")

    print("Loaded train data", train_img.size(), train_cap.size(), train_mask.size())

    DELTA = 10
    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    device = "cuda:0"
    LOSS_FUNCTIONS = {0: teacher_network.RankingLossFunc(DELTA), 1: teacher_network.ContrastiveLoss(1.0, device)}

    train_data = TensorDataset(train_img, train_cap, train_mask)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 10),
    )
    ranking_loss = LOSS_FUNCTIONS[MY_ARGS.loss_function]
    teacher_net.to(device)
    ranking_loss.to(device)

    # optimizer
    params_to_update_share = []

    for name, param in teacher_net.named_parameters():
        if param.requires_grad is True:
            params_to_update_share.append(param)

    params_to_update = list(params_to_update_share)

    optimizer = optim.Adam(params_to_update, lr=0.001)

    print("Start to train")
    queue = None
    for epoch in range(NB_EPOCHS):
        """
        Training
        """

        for step, batch in enumerate(train_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)

            with torch.set_grad_enabled(False):
                img_feature = vision_net.forward(img)
                txt_feature = text_net.forward(cap, mask)

            with torch.set_grad_enabled(True):
                img_vec = teacher_net.forward(img_feature)
                txt_vec = teacher_net.forward(txt_feature)
                txt_vec = txt_vec.detach()
                LOGGER.info("==============before")
                print("img", img_vec[0, :10])
                print("txt", txt_vec[0, :10])

                if queue is None:
                    queue = txt_vec.clone()
                    queue = queue.detach()
                    continue

                loss = ranking_loss(img_vec, txt_vec, queue)
                LOGGER.error("Loss is %f" % loss.item())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                queue = txt_vec.clone()
                queue = queue.detach()

            with torch.set_grad_enabled(False):
                LOGGER.info("============= logits with torch no grad")
                img_vec = teacher_net.forward(img_feature)
                txt_vec = teacher_net.forward(txt_feature)
                logits = ranking_loss.return_logits(img_vec, txt_vec, queue)
                print("logits with queue", logits)
                print(torch.argmax(logits, dim=1))

                print("logits with queue 2")
                preds = ranking_loss.predict(img_vec, queue)
                print(preds)

                print("logits with neg from batch")
                preds = ranking_loss.predict(txt_vec, img_vec)
                print(preds)


                print("img", img_vec[0, :10])
                print("txt", txt_vec[0, :10])
                print()
            break


if __name__ == '__main__':
    main()
