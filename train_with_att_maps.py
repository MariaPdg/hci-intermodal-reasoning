import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import DistilBertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datetime import datetime


def process_batch(id2cap, img2id, _batch, _tokenizer, _prob=0.5):
    _images, _att_maps, _paths = _batch

    for du19 in range(_images.size(0)):
        if random.random() < _prob:
            _images[du19] = torch.mul(_images[du19], _att_maps[du19])

    _paths = utils.preprocess_path(_paths)
    _captions = []
    _masks = []
    longest_length = 0
    for _p in _paths:
        _cap = random.choice([s.rstrip().lower() for s in id2cap[img2id[_p]]])
        _sen = _tokenizer.encode("[CLS] " + _cap + " [SEP]")
        _captions.append(_sen)
        if len(_sen) > longest_length:
            longest_length = len(_sen)
    for _sen in _captions:
        mask = [1] * len(_sen)
        while len(_sen) < longest_length:
            _sen.append(0)
            mask.append(0)
        _masks.append(mask)
        assert len(_sen) == longest_length == len(mask)
    _captions, _masks = torch.from_numpy(np.array(_captions)), torch.from_numpy(np.array(_masks))
    return _images, _captions, _masks


def main(idloss_override=None, att_prob_override=None):
    now = datetime.now()
    logdir = "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    WRITER = SummaryWriter(logdir)
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=50, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=64, type=int)
    PARSER.add_argument("--loss_function", help="which loss function", default=1, type=int)
    PARSER.add_argument("--arch", help="which architecture", default=3, type=int)
    PARSER.add_argument("--optim", help="which optim: adam or sgc", default=1, type=int)
    PARSER.add_argument("--verbose", help="print information", default=1, type=int)
    PARSER.add_argument("--cache", help="if cache the model", default=1, type=int)
    PARSER.add_argument("--end2end", help="if end to end training", default=1, type=int)
    PARSER.add_argument("--idloss", help="if training with id loss", default=0, type=int)
    PARSER.add_argument("--cropping", help="if randomly crop train images", default=1, type=int)
    PARSER.add_argument("--multi", help="if using multi gpu", default=1, type=int)

    MY_ARGS = PARSER.parse_args()
    att_prob = 0.5
    if idloss_override is not None:
        MY_ARGS.idloss = idloss_override
    if att_prob_override is not None:
        att_prob = att_prob_override

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

    BATCH_SIZE = MY_ARGS.batchsize
    NB_EPOCHS = MY_ARGS.epochs
    device = "cuda:0"
    if MY_ARGS.multi > 0:
        device2 = "cuda:1"
    else:
        device2 = "cuda:0"

    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=64, num_workers=2)

    text_net = text_network.TextNet(device2)
    teacher_net2 = teacher_network.TeacherNet3key()
    teacher_net2.to(device2)

    vision_net = vision_network.VisionNet(device)
    teacher_net1 = teacher_network.TeacherNet3query()
    teacher_net1.to(device)

    ranking_loss = teacher_network.ContrastiveLossInBatch(1, device)
    ranking_loss2 = teacher_network.ContrastiveLoss(1, device)
    identification_loss = teacher_network.IdentificationLossInBatch(device)
    teacher_net1.to(device)
    ranking_loss.to(device)
    ranking_loss2.to(device)

    # define if train vision and text net
    if MY_ARGS.end2end != 1:
        text_net.model.eval()
        vision_net.model.eval()
    teacher_net1.train()
    teacher_net2.train()
    ranking_loss.train()

    # optimizer
    if MY_ARGS.optim == 1:
        if MY_ARGS.end2end == 1:
            params = []
            for p in teacher_net1.parameters():
                params.append(p)
            for p in teacher_net2.parameters():
                params.append(p)
            for p in vision_net.parameters():
                params.append(p)
            for p in text_net.parameters():
                params.append(p)
            optimizer = optim.SGD(params,
                                  lr=2e-4, weight_decay=0.0001, momentum=0.9)
            print("Number of training params", utils.calculate_nb_params([teacher_net1, teacher_net2,
                                                                          vision_net, text_net]))
        else:
            optimizer = optim.SGD(teacher_net1.parameters(), lr=2e-4, weight_decay=0.0001, momentum=0.9)

    elif MY_ARGS.optim == 2:
        optimizer = optim.Adam(teacher_net1.parameters(), lr=0.01)

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)

    print("Start to train")
    train_losses = []
    train_accs = []
    train_sim = []
    val_losses = []
    val_accs = []
    val_sim = []

    TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    ID2CAP_TRAIN, IMAGE2ID_TRAIN = utils.read_caption("dataset/annotations/captions_%s2014.json" % "train")
    datasets.ImageFolder.__getitem__ = utils.new_get_att_maps

    if MY_ARGS.cropping == 1:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("dataset/images/train", transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("dataset/images/train", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False)

    for epoch in range(NB_EPOCHS):
        """
        Training
        """
        running_loss = []
        running_loss_total = []
        running_similarity = []
        running_enc1_var = []
        running_enc2_var = []
        running_corrects = 0.0
        total_samples = 0
        teacher_net1.train()
        teacher_net2.train()
        text_net.model.train()
        vision_net.model.train()
        start_time = time.time()

        start_time2 = time.time()
        for step, batch in enumerate(train_loader):
            teacher_net1.train()
            teacher_net2.train()
            text_net.model.train()
            vision_net.model.train()

            img, cap, mask = process_batch(ID2CAP_TRAIN, IMAGE2ID_TRAIN, batch, TOKENIZER, att_prob)
            img, cap, mask = img.to(device), cap.to(device2), mask.to(device2)

            img_feature = vision_net.forward(img)
            txt_feature = text_net.forward(cap, mask)

            img_vec = teacher_net1.forward(img_feature)
            txt_vec = teacher_net2.forward(txt_feature).to(device)

            loss = ranking_loss(img_vec, txt_vec)
            running_loss.append(loss.item())
            if MY_ARGS.idloss:
                loss += identification_loss(img_vec) + identification_loss(txt_vec)
            running_loss_total.append(loss.item())
            loss.backward()

            # update encoder 1 and 2
            optimizer.step()
            optimizer.zero_grad()

            teacher_net1.eval()
            teacher_net2.eval()
            text_net.model.eval()
            vision_net.model.eval()

            with torch.no_grad():
                img_vec = teacher_net1.forward(img_feature)
                txt_vec = teacher_net2.forward(txt_feature).to(device)
                _, preds, avg_similarity = ranking_loss.return_logits(img_vec, txt_vec)
                enc1_var, enc2_var = identification_loss.compute_diff(img_vec), identification_loss.compute_diff(
                    txt_vec)
            running_similarity.append(avg_similarity)
            running_enc1_var.append(enc1_var)
            running_enc2_var.append(enc2_var)

            running_corrects += sum([(0 == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)

        LOGGER.info("Epoch %d: train loss = %f, max=%f min=%f" % (epoch, np.average(running_loss),
                                                                  np.max(running_loss),
                                                                  np.min(running_loss)))
        LOGGER.info(
            "          train acc = %f (%d/%d)" % (
                float(running_corrects / total_samples), running_corrects, total_samples))

        train_losses.append(np.average(running_loss))
        train_accs.append(float(running_corrects / total_samples))
        train_sim.append(np.average(running_similarity))
        WRITER.add_scalar('Loss/train', np.average(running_loss), epoch)
        WRITER.add_scalar('TotalLoss/train', np.average(running_loss_total), epoch)
        WRITER.add_scalar('Accuracy/train', float(running_corrects / total_samples), epoch)
        WRITER.add_scalar('Similarity/train', np.average(running_similarity), epoch)
        WRITER.add_scalar('Var1/train', np.average(running_enc1_var), epoch)
        WRITER.add_scalar('Var2/train', np.average(running_enc2_var), epoch)

        """
        Validating
        """
        running_loss = []
        running_loss_total = []
        running_corrects = 0.0
        total_samples = 0
        running_similarity = []
        running_enc1_var = []
        running_enc2_var = []
        teacher_net1.eval()
        teacher_net2.eval()
        text_net.model.eval()
        vision_net.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valid_dataloader):
                img, cap, mask = batch
                img, cap, mask = img.to(device), cap.to(device2), mask.to(device2)
                img_vec = teacher_net1.forward(vision_net.forward(img))
                txt_vec = teacher_net2.forward(text_net.forward(cap, mask)).to(device)

                loss = ranking_loss(img_vec, txt_vec)
                running_loss.append(loss.item())
                loss += identification_loss(img_vec) + identification_loss(txt_vec)
                running_loss_total.append(loss.item())
                _, preds, avg_similarity = ranking_loss.return_logits(img_vec, txt_vec)
                enc1_var = identification_loss.compute_diff(img_vec)
                enc2_var = identification_loss.compute_diff(txt_vec)

                running_enc1_var.append(enc1_var)
                running_enc2_var.append(enc2_var)
                running_similarity.append(avg_similarity)
                running_corrects += sum([(0 == preds[i]) for i in range(len(preds))])
                total_samples += len(preds)

        LOGGER.info("          val loss = %f, max=%f min=%f" % (np.average(running_loss),
                                                                np.max(running_loss),
                                                                np.min(running_loss)))
        LOGGER.info(
            "          val acc = %f (%d/%d)" % (
                float(running_corrects / total_samples), running_corrects, total_samples))

        val_losses.append(np.average(running_loss))
        val_accs.append(float(running_corrects / total_samples))
        val_sim.append(np.average(running_similarity))
        WRITER.add_scalar('Loss/val', np.average(running_loss), epoch)
        WRITER.add_scalar('TotalLoss/val', np.average(running_loss_total), epoch)
        WRITER.add_scalar('Accuracy/val', float(running_corrects / total_samples), epoch)
        WRITER.add_scalar('Similarity/val', np.average(running_similarity), epoch)
        WRITER.add_scalar('Var1/val', np.average(running_enc1_var), epoch)
        WRITER.add_scalar('Var2/val', np.average(running_enc2_var), epoch)

        start_time3 = time.time()
        LOGGER.error("Training took %.3f (aug: %.3f, compute: %.3f)" % (start_time3 - start_time,
                                                                        start_time2 - start_time,
                                                                        start_time3 - start_time2))

    if MY_ARGS.cache == 1:
        torch.save(teacher_net1.state_dict(), "models/enc1-t1-%s" % now.strftime("%Y%m%d-%H%M%S"))
        torch.save(teacher_net2.state_dict(), "models/enc2-t2-%s" % now.strftime("%Y%m%d-%H%M%S"))
        torch.save(vision_net.model.state_dict(), "models/enc1-%s" % now.strftime("%Y%m%d-%H%M%S"))
        torch.save(text_net.model.state_dict(), "models/enc2-%s" % now.strftime("%Y%m%d-%H%M%S"))

    WRITER.close()

    # plotting
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.dpi"] = 200

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(range(len(train_losses)), train_losses,
                range(len(train_losses)), val_losses, '-')
    axs[0].set_title('loss')
    fig.suptitle('Training loss and accuracy with batch size %d' % BATCH_SIZE, fontsize=16)

    axs[1].plot(range(len(train_accs)), train_accs,
                range(len(val_accs)), val_accs, '-')
    axs[1].set_xlabel('epoch')
    axs[1].set_title('acc')

    axs[2].plot(range(len(train_sim)), train_sim,
                range(len(val_accs)), val_sim, '-')
    axs[2].set_xlabel('epoch')
    axs[2].set_title('sim')

    fig_dir = "figures/fig_training2enc-arch%d-optim%d-nogradclip.png" % (MY_ARGS.arch,
                                                                          MY_ARGS.optim)
    fig.savefig(fig_dir)
    print("plotting figures save at %s" % fig_dir)

    text_net.model.cpu()
    teacher_net2.cpu()
    vision_net.model.cpu()
    teacher_net1.cpu()
    img.cpu()
    cap.cpu()
    mask.cpu()
    img_vec.cpu()
    txt_vec.cpu()

    del ranking_loss
    del ranking_loss2
    del identification_loss
    del text_net
    del vision_net
    del teacher_net1
    del teacher_net2

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
