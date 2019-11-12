import json
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import termcolor
import sys
import gc
import time
import threading
import multiprocessing
import pickle
from transformers import DistilBertTokenizer
from knockknock import slack_sender


sys.stdin = open("webhook_url.txt", "r")
SLACK_WEBHOOK = sys.stdin.readline().rstrip()


class Logger:
    def __init__(self):
        return

    def info(self, information):
        print(termcolor.colored("[INFO] %s" % information, "green", attrs=["bold"]))

    def error(self, information):
        print(termcolor.colored("[ERROR] %s" % information, "red", attrs=["bold"]))


def read_caption(filename="dataset/annotations/captions_val2014.json"):
    with open(filename) as json_file:
        data = json.load(json_file)

        id2cap = {}
        for ann in data["annotations"]:
            if ann["image_id"] not in id2cap:
                id2cap[ann["image_id"]] = [ann["caption"]]
            else:
                id2cap[ann["image_id"]].append(ann["caption"])

        filename2id = {}
        for img in data["images"]:
            assert img["file_name"] not in filename2id
            filename2id[img["file_name"]] = img["id"]

    return id2cap, filename2id


def preprocess_path(paths):
    return list(map(lambda x: x.split("/")[-1], paths))


def new_get(self, index):
    path, _ = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    return sample, path


def cache_data_helper1(which, limit, return_dict):
        ID2CAP, IMAGE2ID = read_caption("dataset/annotations/captions_%s2014.json" % which)
        traindir = "dataset/%s" % which
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        datasets.ImageFolder.__getitem__ = new_get
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True)

        images = []
        texts = []
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        longest_length = 0
        print("caching data")

        for step, batch in enumerate(train_loader):
            image, cap = batch[0][0], ID2CAP[IMAGE2ID[preprocess_path(batch[1])[0]]][0]
            sen = tokenizer.encode("[CLS] " + cap + " [SEP]")
            if len(sen) > longest_length:
                longest_length = len(sen)
            images.append(image)
            texts.append(sen)
            if step > limit > 0:
                break
        print("start to save")
        images = torch.stack(images)
        torch.save(images, "cached_data/%s_img" % which)
        with open('cached_data/%s_text' % which, 'wb') as fp:
            pickle.dump(texts, fp)
        print(images.size(), longest_length)
        return_dict["l"] = longest_length


def cache_data_helper2(which, longest_length):
    with open('cached_data/%s_text' % which, 'rb') as fp:
        texts = pickle.load(fp)
    masks = []
<<<<<<< HEAD
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    longest_length = 0
    print("caching data")
    for step, batch in enumerate(train_loader):
        image, cap = batch[0][0], ID2CAP[IMAGE2ID[preprocess_path(batch[1])[0]]][0]
        sen = tokenizer.encode("[CLS] " + cap + " [SEP]")
        if len(sen) > longest_length:
            longest_length = len(sen)
        images.append(image)
        texts.append(sen)
        if step > limit > 0:
            break
    
    print("start to save")
    images = torch.stack(images)
    torch.save(images, "cached_data/%s_img" % which)
    print(images.size())

    time.sleep(5)
    del images, train_loader, batch, image, cap
    gc.collect()
    print("free done")
    time.sleep(5)
    
    print("begin padding")
=======
    print("begin padding with %d" % longest_length)
>>>>>>> cce77c0ae168129811425ebca01b034c447d22da
    for sample in texts:
        mask = [1] * len(sample)
        while len(sample) < longest_length:
            sample.append(0)
            mask.append(0)
        masks.append(mask)
        assert len(sample) == longest_length == len(mask)
    texts, masks = torch.from_numpy(np.array(texts)), torch.from_numpy(np.array(masks))

    print(texts.size(), masks.size())
    torch.save(texts, "cached_data/%s_cap" % which)
    torch.save(masks, "cached_data/%s_mask" % which)


@slack_sender(webhook_url=SLACK_WEBHOOK, channel="bot")
def cache_data(which="val", limit=5):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p1 = multiprocessing.Process(target=cache_data_helper1, args=(which, limit, return_dict))
    p1.start()
    p1.join()
    print("step 1 is done")
    time.sleep(5)

    cache_data_helper2(which, return_dict["l"])


if __name__ == "__main__":
    cache_data("train", limit=1000)
    #cache_data("val", limit=-1)

