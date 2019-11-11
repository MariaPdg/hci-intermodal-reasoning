import json
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import DistilBertTokenizer


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


def cache_data():
    ID2CAP, IMAGE2ID = read_caption()
    traindir = "dataset/val"
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
        batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True)

    images = []
    texts = []
    masks = []
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    longest_length = 0
    for step, batch in enumerate(train_loader):
        image, cap = batch[0][0], ID2CAP[IMAGE2ID[preprocess_path(batch[1])[0]]][0]
        sen = tokenizer.encode("[CLS] " + cap + " [SEP]")
        if len(sen) > longest_length:
            longest_length = len(sen)
        images.append(image)
        texts.append(sen)

        if step > 3:
            break

    for sample in texts:
        mask = [1] * len(sample)
        while len(sample) < longest_length:
            sample.append(0)
            mask.append(0)
        masks.append(mask)
        assert len(sample) == longest_length == len(mask)
    texts, masks = torch.from_numpy(np.array(texts)), torch.from_numpy(np.array(masks))
    images = torch.stack(images)

    print(images.size(), texts.size(), masks.size())
    torch.save(images, "cached_data/val_img")
    torch.save(texts, "cached_data/val_cap")
    torch.save(masks, "cached_data/val_mask")


if __name__ == "__main__":
    # read_caption()
    cache_data()

