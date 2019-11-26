#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import types
import torch.nn as nn


# In[15]:


class Vis_Net:

    def __init__(self, dev="cpu"):
        #self.model = models.alexnet(pretrained=True)
        #self.model.to(dev)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            #nn.Linear(4096, 10000),
        )
    
        #num_ftrs = self.model.classifier[6].in_features
        #self.model.classifier[6] = nn.Linear(num_ftrs, 9216)  # reinitialize the 6th layer


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
#     def parameters(self):
#         return self.model.parameters()

#     def named_parameters(self):
#         return self.model.named_parameters()
    


# In[16]:


if __name__ == "__main__":
    
    net = Vis_Net()
    traindir = "../hci-intermodal-reasoning/dataset/val"
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

    


# In[ ]:




