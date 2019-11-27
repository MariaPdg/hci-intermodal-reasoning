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
    PARSER.add_argument("--epochs", help="number of epochs", default=10, type=int)
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

    val_img = torch.load("cached_data/val_img")
    val_cap = torch.load("cached_data/val_cap")
    val_mask = torch.load("cached_data/val_mask")

    print("Loaded train data", train_img.size(), train_cap.size(), train_mask.size())
    print("Loaded val data", val_img.size(), val_cap.size(), val_mask.size())

    DELTA = 10
    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    train_modality_net = bool(MY_ARGS.train_modality_net)
    device = "cuda:0"
    verbose = bool(MY_ARGS.verbose)
    LOSS_FUNCTIONS = {0: teacher_network.RankingLossFunc(DELTA), 1: teacher_network.ContrastiveLoss(10, device)}

    train_data = TensorDataset(train_img, train_cap, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=2)
    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=2)

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net = teacher_network.TeacherNet()
    ranking_loss = LOSS_FUNCTIONS[MY_ARGS.loss_function]
    teacher_net.to(device)
    ranking_loss.to(device)

    # optimizer
    params_to_update_share = []

    for name, param in teacher_net.named_parameters():
        if param.requires_grad is True:
            params_to_update_share.append(param)

    params_to_update = list(params_to_update_share)

    optimizer = optim.Adam(params_to_update, lr=0.0001)

    print("Start to train")
    start_time = time.time()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    queue = None
    for epoch in range(NB_EPOCHS):
        """
        Training
        """
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
                txt_vec = txt_vec.detach()

                if queue is None:
                    queue = txt_vec.clone()
                    queue = queue.detach()
                    continue

                loss = ranking_loss(img_vec, txt_vec, queue)
                try:
                    optimizer.zero_grad()
                    loss.backward()
                except AttributeError:
                    print("step %d faulty" % step)
                    continue

                optimizer.step()
                queue = txt_vec.clone()
                queue = queue.detach()

            with torch.set_grad_enabled(True):
                preds = ranking_loss.predict(img_vec, txt_vec)

            try:
                running_loss += loss.item()
            except AttributeError:
                pass
            running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)

        if verbose:
            LOGGER.info("Epoch %d: train loss = %f" % (epoch, running_loss/step))
            LOGGER.info(
                "          train acc = %f (%d/%d)" % (
                    float(running_corrects / total_samples), running_corrects, total_samples))

        train_losses.append(running_loss/step)
        train_accs.append(float(running_corrects / total_samples))

        """
        Validating
        """
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0

        for step, batch in enumerate(valid_dataloader):
            img, cap, mask = tuple(t.to(device) for t in batch)

            with torch.set_grad_enabled(False):
                img_vec = teacher_net.forward(vision_net.forward(img))
                txt_vec = teacher_net.forward(text_net.forward(cap, mask))

                loss = ranking_loss(img_vec, txt_vec, queue)
                preds = ranking_loss.predict(img_vec, txt_vec)

                queue = txt_vec.clone()

            running_loss += loss.item()
            running_corrects += sum([(i == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)

        if verbose or epoch == NB_EPOCHS - 1:
            LOGGER.info("Val loss = %f" % (running_loss/step))
            LOGGER.info(
                "Val acc = %f (%d/%d)" % (float(running_corrects / total_samples), running_corrects, total_samples))
        val_losses.append(running_loss/step)
        val_accs.append(float(running_corrects / total_samples))

    LOGGER.info("Training done in %f mins" % ((time.time() - start_time) / 60))

    model_name = "%d-train_modality%d" % (running_corrects, train_modality_net)
    torch.save(teacher_net.state_dict(), "models/%s" % model_name)
    print(train_losses)
    print(train_accs)
    print(val_losses)
    print(val_accs)
    print()
    return val_accs[-1], MY_ARGS


if __name__ == '__main__':
    main()
