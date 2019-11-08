import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


class VisionNet:
    def __init__(self):
        self.model = torchvision.models.resnext101_32x8d(pretrained=True)
        print(self.model.state_dict)

    def forward(self, image):
        return self.model.forward_layer(image)


if __name__ == "__main__":
    net = VisionNet()

    traindir = "dataset/val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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
        out = net.forward(batch[0])
        print(out.size())
        break
