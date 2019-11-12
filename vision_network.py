import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import types


def forward_layer(self, y):
    y = self.conv1(y)
    y = self.bn1(y)
    y = self.relu(y)
    y = self.maxpool(y)

    y = self.layer1(y)
    y = self.layer2(y)
    y = self.layer3(y)
    y = self.layer4(y)

    y = self.avgpool(y)
    y = torch.flatten(y, 1)

    return y


class VisionNet:
    def __init__(self, dev="cpu"):
        self.model = torchvision.models.resnext101_32x8d(pretrained=True)
        self.model.forward_layer = types.MethodType(forward_layer, self.model)
        self.model.to(dev)

    def forward(self, image):
        return self.model.forward_layer(image)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()


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
        print(batch[0].size())
        out = net.forward(batch[0])
        print(out.size())
        break
