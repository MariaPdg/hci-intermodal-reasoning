#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import gensim
import types

from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler



# In[ ]:


# class Text_Net(nn.Module):

#     def __init__(self):
#         super(Text_Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=16, out_channels=300, kernel_size=3)
#         self.conv2 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
#         self.conv3 = nn.Conv1d(in_channels=300, out_channels=300, kernel_size=3)
#         self.fc1 = nn.Linear(in_features=300*73, out_features=9216)

#     def forward(self, inputs):
#         out = F.relu(self.conv1(inputs))
#         out = F.max_pool1d(F.relu(self.conv2(out)), kernel_size=2)
#         out = F.max_pool1d(F.relu(self.conv3(out)), kernel_size=2)
#         out = self.fc1(out.view(-1, 300*73))
#         return out

class Text_Net(nn.Module):

    def __init__(self):
        super(Text_Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=52, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=52, out_channels=52, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=52, out_channels=52, kernel_size=3)
        self.fc1 = nn.Linear(in_features=52*73, out_features=9216)

    def forward(self, inputs):
        out = F.relu(self.conv1(inputs))
        out = F.max_pool1d(F.relu(self.conv2(out)), kernel_size=2)
        out = F.max_pool1d(F.relu(self.conv3(out)), kernel_size=2)
        out = self.fc1(out.view(-1, 52*73))
        return out



