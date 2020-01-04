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
import os
import random
import matplotlib.pyplot as plt
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from knockknock import slack_sender
from transformers import DistilBertTokenizer
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


def sample_neg_vectors(_neg_space, _positive_img_id, _postive_img_vec, id2cap, _tokenizer, _text_model_func,
                       device="cpu", _nb_neg_vectors=63):
    """

    :param _neg_space:
    :param _positive_img_id:
    :param id2cap:
    :param _tokenizer:
    :param _text_model_func: must be in eval mode
    :return:
    """
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
        _sen = _tokenizer.encode("[CLS] " + _cap + " [SEP]")
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
    _neg_sampler = SequentialSampler(_neg_data)
    _neg_dataloader = DataLoader(_neg_data, sampler=_neg_sampler, batch_size=100, num_workers=2)
    _neg_tracker = []
    for _batch in _neg_dataloader:
        _du1, _du2 = tuple(t.to(device) for t in _batch)
        _neg_vec = _text_model_func(_du1, _du2)
        for _i in range(_neg_vec.size(0)):
            _score = torch.matmul(_postive_img_vec.view(1, 100), _neg_vec[_i].view(100, 1)).item()
            _neg_tracker.append((_score, _du1[_i], _du2[_i]))
    _res = [_neg_tracker[dummy2] for dummy2 in np.argsort([dummy[0] for dummy in _neg_tracker])[-_nb_neg_vectors:]]
    _neg_cap = torch.cat([dummy3[1].view(1, dummy3[1].size(0)) for dummy3 in _res], dim=0)
    _neg_mask = torch.cat([dummy3[2].view(1, dummy3[2].size(0)) for dummy3 in _res], dim=0)
    return _neg_cap, _neg_mask


# ID2CAP_TRAIN, IMAGE2ID_TRAIN = utils.read_caption("dataset/annotations/captions_%s2014.json" % "train")
#
# with open('cached_data/id2cap_train.json', 'wb') as fp:
#     pickle.dump(ID2CAP_TRAIN, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open('cached_data/image2id_train.json', 'wb') as fp:
#     pickle.dump(IMAGE2ID_TRAIN, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('cached_data/id2cap_train.json', 'rb') as fp:
    ID2CAP_TRAIN = pickle.load(fp)
with open('cached_data/image2id_train.json', 'rb') as fp:
    IMAGE2ID_TRAIN = pickle.load(fp)

IMAGES_LIST = list(IMAGE2ID_TRAIN.values())
random.shuffle(IMAGES_LIST)
CHUNKS = np.array_split(IMAGES_LIST, 100)
datasets.ImageFolder.__getitem__ = utils.new_get
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder("dataset/images/train", transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])),
    batch_size=64, shuffle=False,
    num_workers=2, pin_memory=False)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

for epoch in range(250):
    for step, batch in enumerate(train_loader):
        im, cap, mask, id_code = process_batch(ID2CAP_TRAIN, IMAGE2ID_TRAIN, batch, tokenizer)
        NEG_SPACE = CHUNKS[epoch % len(CHUNKS)]
        pos_im_vec = fake_img_func(im)
        for index in range(im.size(0)):
            neg_cap, neg_mask = sample_neg_vectors(NEG_SPACE, id_code[index], pos_im_vec[index], ID2CAP_TRAIN, tokenizer, fake_text_func, "cuda:0")
            print(neg_cap.size(), neg_mask.size())
        break
    break
