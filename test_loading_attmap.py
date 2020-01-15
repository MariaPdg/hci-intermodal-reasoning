import torch
import pickle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import utils


def new_get_att_maps(self, index):
    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    path, _ = self.samples[index]
    att_path = path.replace("train14/", "").replace("images", "maps").replace(".jpg", ".png")
    _att_map = Image.open(att_path)
    sample = self.loader(path)

    torch.manual_seed(seed)
    random.seed(seed)
    if self.transform is not None:
        sample = self.transform(sample)
        sample = norm_transform(sample)

    torch.manual_seed(seed)
    random.seed(seed)
    if self.transform is not None:
        _att_map = self.transform(_att_map)

    return sample, _att_map, path


datasets.ImageFolder.__getitem__ = new_get_att_maps


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder("dataset/images/train", transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])),
    batch_size=4, shuffle=True,
    num_workers=2, pin_memory=False)

plt.figure(figsize=(20, 20))

for i, batch in enumerate(train_loader):
    img, att_map, _ = batch
    plt.subplot(4, 1, 1)
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(img, padding=5, normalize=False, pad_value=255),
                            (1, 2, 0)))

    plt.subplot(4, 1, 2)
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(att_map, padding=5, normalize=False, pad_value=255),
                            (1, 2, 0)))
    plt.subplot(4, 1, 3)
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(torch.mul(img, att_map), padding=5, normalize=False, pad_value=255),
                            (1, 2, 0)))

    plt.subplot(4, 1, 4)
    plt.axis("off")
    print(img.size(), att_map.size(), img[:, 0, :, :].size())
    img[:, 0, :, :] = torch.mul(img[:, 0, :, :], att_map.squeeze())
    img[:, 1, :, :] = torch.mul(img[:, 1, :, :], att_map.squeeze())
    img[:, 2, :, :] = torch.mul(img[:, 2, :, :], att_map.squeeze())

    plt.imshow(np.transpose(vutils.make_grid(torch.mul(img, att_map), padding=5, normalize=False, pad_value=255),
                            (1, 2, 0)))

    plt.tight_layout()
    plt.savefig("temp/fig.png")
    break

