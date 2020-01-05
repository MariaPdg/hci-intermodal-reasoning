import torch
import utils
import text_network
import teacher_network
import vision_network
import torch.optim as optim
import time
import pickle
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import GPUtil
import sys
from transformers import DistilBertTokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datetime import datetime


def preprocess_captions(_captions_list):
    res = []
    for _cap in _captions_list:
        res.append(_cap.rstrip().lower())
    return res


def fake_img_func(inp1):
    return torch.rand(inp1.size(0), 100)


def fake_text_func(inp1, inp2):
    return torch.rand(inp1.size(0), 100)


def process_batch(id2cap, img2id, _batch, _tokenizer):
    _images, _paths = _batch
    _paths = utils.preprocess_path(_paths)
    _captions = []
    _masks = []
    _id = []
    longest_length = 0
    for _p in _paths:
        _id.append(img2id[_p])
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
    return _images, _captions, _masks, _id


def forward_neg_space(_neg_space, _text2vec, id2cap, _text_model_func, _tokenizer, device):
    _neg_cap_list = []
    for _neg_img_id in _neg_space:
        _neg_cap_list.extend(id2cap[_neg_img_id])
    _neg_cap_list = preprocess_captions(_neg_cap_list)
    _neg_cap_tokenized_list = []
    _neg_masks = []
    longest_len = 0
    _cap2vec = {}
    for _cap in _neg_cap_list:
        try:
            _sen = _text2vec[_cap]
        except KeyError:
            _sen = _tokenizer.encode("[CLS] " + _cap + " [SEP]")
            _text2vec[_cap] = _sen
        _neg_cap_tokenized_list.append(_sen)
        if len(_sen) > longest_len:
            longest_len = len(_sen)
    for _sen in _neg_cap_tokenized_list:
        _mask = [1] * len(_sen)
        while len(_sen) < longest_len:
            _sen.append(0)
            _mask.append(0)
        _neg_masks.append(_mask)
        assert len(_sen) == longest_len == len(_mask)

    _captions, _masks = torch.from_numpy(np.array(_neg_cap_tokenized_list)), torch.from_numpy(np.array(_neg_masks))
    _neg_data = TensorDataset(_captions, _masks, torch.arange(_captions.size(0)))
    _neg_sampler = SequentialSampler(_neg_data)  # must not be random here
    _neg_dataloader = DataLoader(_neg_data, sampler=_neg_sampler, batch_size=32, num_workers=5)

    with torch.no_grad():
        for _step, _batch in enumerate(_neg_dataloader):
            _du1, _du2 = tuple(t.to(device) for t in _batch[:2])
            _neg_vec = _text_model_func(_du1, _du2)
            for _idx2, _idx in enumerate(_batch[2]):
                _cap2vec[_neg_cap_list[_idx.item()]] = _neg_vec[_idx2]
    del _du1
    del _du2
    del _neg_vec
    torch.cuda.empty_cache()
    return _cap2vec


def sample_neg_vectors(_neg_space, _positive_img_id, _postive_img_vec, id2cap, _tokenizer, _text2vec, _cap2vec,
                       _text_model_func,
                       device="cpu", _nb_neg_vectors=63, clear_mem=True):
    _neg_space = list(_neg_space)
    try:
        _neg_space.remove(_positive_img_id)
    except ValueError:
        pass

    _neg_cap_list = []
    for _neg_img_id in _neg_space:
        _neg_cap_list.extend(id2cap[_neg_img_id])
    _neg_cap_list = preprocess_captions(_neg_cap_list)
    _neg_cap_tokenized_list = []
    _neg_masks = []
    longest_len = 0
    for _cap in _neg_cap_list:
        try:
            _sen = _text2vec[_cap]
        except KeyError:
            _sen = _tokenizer.encode("[CLS] " + _cap + " [SEP]")
            _text2vec[_cap] = _sen
        _neg_cap_tokenized_list.append(_sen)
        if len(_sen) > longest_len:
            longest_len = len(_sen)
    for _sen in _neg_cap_tokenized_list:
        _mask = [1] * len(_sen)
        while len(_sen) < longest_len:
            _sen.append(0)
            _mask.append(0)
        _neg_masks.append(_mask)
        assert len(_sen) == longest_len == len(_mask)

    _captions, _masks = torch.from_numpy(np.array(_neg_cap_tokenized_list)), torch.from_numpy(np.array(_neg_masks))
    _neg_data = TensorDataset(_captions, _masks)
    _neg_sampler = SequentialSampler(_neg_data)  # must not be random here
    _neg_dataloader = DataLoader(_neg_data, sampler=_neg_sampler, batch_size=int(_captions.size(0)/10), num_workers=5)
    _neg_tracker = []
    _all_scores = []

    for _cap in _neg_cap_list:
        _neg_vec = _cap2vec[_cap]
        _score = torch.matmul(_postive_img_vec.view(1, 100), _neg_vec.view(100, 1)).item()
        _all_scores.append(_score)

    _res = np.argsort(_all_scores)[-_nb_neg_vectors:]

    _neg_cap = _captions[_res]
    _neg_mask = _masks[_res]

    return _neg_cap.to(device), _neg_mask.to(device)


def tokenize_neg_space(_neg_spaces, id2cap, _tokenizer):
    _all = []
    for _neg_space in _neg_spaces:
        _res = {}
        _neg_cap_list = []
        for _neg_img_id in _neg_space:
            _neg_cap_list.extend(id2cap[_neg_img_id])
        _neg_cap_list = preprocess_captions(_neg_cap_list)
        for _cap in _neg_cap_list:
            if _cap not in _res:
                _sen = _tokenizer.encode("[CLS] " + _cap + " [SEP]")
                _res[_cap] = _sen
        _all.append(_res)
    return _all


def main(idloss_override=None):
    LOGGER = utils.Logger()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--epochs", help="number of epochs", default=10, type=int)
    PARSER.add_argument("--batchsize", help="batch size", default=16, type=int)
    PARSER.add_argument("--loss_function", help="which loss function", default=1, type=int)
    PARSER.add_argument("--arch", help="which architecture", default=3, type=int)
    PARSER.add_argument("--optim", help="which optim: adam or sgc", default=1, type=int)
    PARSER.add_argument("--verbose", help="print information", default=1, type=int)
    PARSER.add_argument("--cache", help="if cache the model", default=0, type=int)
    PARSER.add_argument("--end2end", help="if end to end training", default=1, type=int)
    PARSER.add_argument("--idloss", help="if training with id loss", default=0, type=int)
    PARSER.add_argument("--cropping", help="if randomly crop train images", default=1, type=int)
    PARSER.add_argument("--debug", help="if debugging", default=1, type=int)

    MY_ARGS = PARSER.parse_args()

    now = datetime.now()
    if MY_ARGS.debug == 0:
        logdir = "logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    else:
        logdir = "logs/debug/"
    WRITER = SummaryWriter(logdir)

    if idloss_override is not None:
        MY_ARGS.idloss = idloss_override

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

    valid_data = TensorDataset(val_img, val_cap, val_mask)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=2)

    text_net = text_network.TextNet(device)
    vision_net = vision_network.VisionNet(device)
    teacher_net1 = teacher_network.TeacherNet3query()
    teacher_net2 = teacher_network.TeacherNet3key()
    ranking_loss = teacher_network.ContrastiveLossReRank(1, device)
    ranking_loss2 = teacher_network.ContrastiveLossInBatch(1, device)
    identification_loss = teacher_network.IdentificationLossInBatch(device)
    teacher_net1.to(device)
    teacher_net2.to(device)
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
    datasets.ImageFolder.__getitem__ = utils.new_get

    with open('cached_data/id2cap_train.json', 'rb') as fp:
        ID2CAP_TRAIN = pickle.load(fp)
    with open('cached_data/image2id_train.json', 'rb') as fp:
        IMAGE2ID_TRAIN = pickle.load(fp)

    def text_func(inp1, inp2):
        something = text_net.forward(inp1, inp2)
        return teacher_net2.forward(something)

    IMAGES_LIST = list(IMAGE2ID_TRAIN.values())
    random.shuffle(IMAGES_LIST)
    CHUNKS = np.array_split(IMAGES_LIST, 100)
    TEXT2VEC_ALL = tokenize_neg_space(CHUNKS, ID2CAP_TRAIN, TOKENIZER)

    if MY_ARGS.cropping == 1:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("dataset/images/train", transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("dataset/images/train", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
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
        NEG_SPACE = CHUNKS[epoch % len(CHUNKS)]
        TEXT2VEC = TEXT2VEC_ALL[epoch % len(CHUNKS)]
        CAP2VEC = forward_neg_space(NEG_SPACE, TEXT2VEC, ID2CAP_TRAIN, text_func, TOKENIZER, device)
        start_time = time.time()

        start_time2 = time.time()
        for step, batch in enumerate(train_loader):
            teacher_net1.eval()
            teacher_net2.eval()
            text_net.model.eval()
            vision_net.model.eval()

            st1 = time.time()
            with torch.no_grad():
                img, cap, mask, id_code = process_batch(ID2CAP_TRAIN, IMAGE2ID_TRAIN, batch, TOKENIZER)
                img, cap, mask = tuple(t.to(device) for t in (img, cap, mask))

                img_vec = teacher_net1.forward(vision_net.forward(img))
                neg_samples = []

                for index in range(img_vec.size(0)):
                    neg_cap, neg_mask = sample_neg_vectors(NEG_SPACE, id_code[index], img_vec[index], ID2CAP_TRAIN,
                                                           TOKENIZER,
                                                           TEXT2VEC, CAP2VEC,
                                                           text_func,
                                                           device, 10, index == img_vec.size(0)-1)
                    neg_samples.append((neg_cap, neg_mask))
            st2 = time.time()

            teacher_net1.train()
            teacher_net2.train()
            text_net.model.train()
            vision_net.model.train()

            img_vec = teacher_net1.forward(vision_net.forward(img))
            pos_txt_vec = teacher_net2.forward(text_net.forward(cap, mask))

            neg_txt_vecs = [teacher_net2.forward(text_net.forward(sample[0], sample[1])) for sample in neg_samples]

            loss = ranking_loss(img_vec, pos_txt_vec, neg_txt_vecs)
            running_loss.append(loss.item())
            if MY_ARGS.idloss:
                loss += identification_loss(img_vec) + identification_loss(txt_vec)
            running_loss_total.append(loss.item())
            loss.backward()
            st3 = time.time()

            # update encoder 1 and 2
            optimizer.step()
            optimizer.zero_grad()

            teacher_net1.eval()
            teacher_net2.eval()
            text_net.model.eval()
            vision_net.model.eval()

            img_vec = teacher_net1.forward(vision_net.forward(img))
            txt_vec = teacher_net2.forward(text_net.forward(cap, mask))
            _, preds, avg_similarity = ranking_loss.return_logits(img_vec, txt_vec, neg_txt_vecs)
            enc1_var, enc2_var = identification_loss.compute_diff(img_vec), identification_loss.compute_diff(txt_vec)
            running_similarity.append(avg_similarity)
            running_enc1_var.append(enc1_var)
            running_enc2_var.append(enc2_var)

            running_corrects += sum([(0 == preds[i]) for i in range(len(preds))])
            total_samples += len(preds)
            st4 = time.time()
            # print("1 step took %.3f, sampling took %.3f, forwarding took %.3f, updating took %.3f" % (time.time()-st1, st2-st1, st3-st2, st4-st3))

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
                img, cap, mask = tuple(t.to(device) for t in batch)
                img_vec = teacher_net1.forward(vision_net.forward(img))
                txt_vec = teacher_net2.forward(text_net.forward(cap, mask))

                loss = ranking_loss2(img_vec, txt_vec)
                running_loss.append(loss.item())
                loss += identification_loss(img_vec) + identification_loss(txt_vec)
                running_loss_total.append(loss.item())
                _, preds, avg_similarity = ranking_loss2.return_logits(img_vec, txt_vec)
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
        LOGGER.error("Training took %.3f (aug: %.3f, compute: %.3f)" % (start_time3-start_time,
                                                                        start_time2-start_time,
                                                                        start_time3-start_time2))

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


if __name__ == '__main__':
    main()
