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


def main():
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=10, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=3, type=int)
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

    DELTA = 0.002
    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    train_modality_net = bool(MY_ARGS.train_modality_net)
    device = "cuda:0"
    verbose = bool(MY_ARGS.verbose)
    LOSS_FUNCTIONS = {0: teacher_network.RankingLossFunc(DELTA), 1: teacher_network.ContrastiveLoss(DELTA, device)}

    train_data = TensorDataset(train_img, train_cap, train_mask)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=2)

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 8),
        torch.nn.Softmax(dim=1),
    )
    ranking_loss = LOSS_FUNCTIONS[MY_ARGS.loss_function]
    teacher_net.to(device)
    ranking_loss.to(device)

    # optimizer
    params_to_update_share = []
    params_to_update_img = []
    params_to_update_txt = []

    for name, param in teacher_net.named_parameters():
        if param.requires_grad is True:
            params_to_update_share.append(param)

    for name, param in vision_net.named_parameters():
        if param.requires_grad is True:
            params_to_update_img.append(param)

    for name, param in text_net.named_parameters():
        if param.requires_grad is True:
            params_to_update_txt.append(param)

    params_to_update = list(params_to_update_share) + list(params_to_update_img) + list(params_to_update_txt)
    optimizer = optim.Adam(params_to_update, lr=0.01)

    print("Start to train")
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    queue = None
    for epoch in range(NB_EPOCHS):
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)

            with torch.set_grad_enabled(train_modality_net):
                img_feature = vision_net.forward(img)
                txt_feature = text_net.forward(cap, mask)
                txt_vec = teacher_net.forward(txt_feature)

            with torch.set_grad_enabled(True):
                img_vec = teacher_net.forward(img_feature)
                #                 print("before", torch.max(img_vec), torch.max(txt_vec))
                print("before", img_vec[:, :10])
                print(txt_vec[:, :10])

                loss = ranking_loss(img_vec, txt_vec)
                print(loss)
                preds = ranking_loss.predict(img_vec, txt_vec)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                img_vec = teacher_net.forward(img_feature)
                txt_vec = teacher_net.forward(txt_feature)
                #                 print("after", torch.max(img_vec), torch.max(txt_vec))
                print("after", img_vec[:, :10])
                print(txt_vec[:, :10])

                print()

            running_loss += loss.item() * BATCH_SIZE
            running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)
            break


if __name__ == '__main__':
    main()