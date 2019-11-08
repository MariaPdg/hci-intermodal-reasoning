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


def caption2index(id2cap, samples_to_load=-1):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    indices = []
    longest_length = 0
    for cap in id2cap.values():
        sen = tokenizer.encode("[CLS] "+cap+" [SEP]")
        indices.append(sen)
        if len(sen) > longest_length:
            longest_length = len(sen)

        if len(indices) > samples_to_load > 0:
            break
    return indices, longest_length


def index2tensor(indices, longest_length):
    masks = []
    for sample in indices:
        mask = [1] * len(sample)
        while len(sample) < longest_length:
            sample.append(0)
            mask.append(0)
        masks.append(mask)
        assert len(sample) == longest_length == len(mask)
    return torch.from_numpy(np.array(indices)), torch.from_numpy(np.array(masks))


if __name__ == "__main__":
    ID2CAP, _ = read_caption()
    i, l = caption2index(ID2CAP, 100)
    tensor, tensor2 = index2tensor(i, l)
    print(tensor.size())
    torch.save(tensor, "cached_data/val_cap")
    torch.save(tensor2, "cached_data/val_mask")

