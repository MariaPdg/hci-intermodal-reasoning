import json
import torch
import numpy as np
from transformers import DistilBertTokenizer


def read_caption(filename="dataset/annotations/captions_val2014.json"):
    with open(filename) as json_file:
        data = json.load(json_file)

        id2cap = {}
        for ann in data["annotations"]:
            if ann["id"] not in id2cap:
                id2cap[ann["id"]] = ann["caption"]
            else:
                id2cap[ann["id"]].append(ann["caption"])

        filename2id = {}
        for img in data["images"]:
            assert img["file_name"] not in filename2id
            filename2id[img["file_name"]] = img["id"]

    return id2cap, filename2id


def caption2index(id2cap):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    indices = []
    longest_length = 0
    for cap in id2cap.values():
        sen = tokenizer.encode("[CLS] "+cap+" [SEP]")
        indices.append(sen)
        if len(sen) > longest_length:
            longest_length = len(sen)
    return indices, longest_length


def index2tensor(indices, longest_length):
    for sample in indices:
        while len(sample) < longest_length:
            sample.append(0)
        assert len(sample) == longest_length
    return torch.from_numpy(np.array(indices))


if __name__ == "__main__":
    ID2CAP, _ = read_caption()
    i, l = caption2index(ID2CAP)
    tensor = index2tensor(i, l)
    print(tensor.size())
