import torchvision.transforms as transforms
import torchvision.datasets as datasets
import types
import torch
import utils
import text_network
import teacher_network
import vision_network


ID2CAP, IMAGE2ID = utils.read_caption()
traindir = "dataset/val"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def new_get(self, index):
    path, _ = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    return sample, path


datasets.ImageFolder.__getitem__ = new_get
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=2, shuffle=True,
    num_workers=2, pin_memory=True)

for step, batch in enumerate(train_loader):
    print(batch[0].size())
    print(utils.preprocess_path(batch[1]))
    break

