#!/usr/bin/env python
# coding: utf-8

# In[104]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models


# In[105]:


print("Torch Version: ", torch.__version__)


# In[106]:


class Image_Net():

    def __init__(self, dev="cpu"):
        super(Image_Net, self).__init__()

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, feature_extract, use_pretrained=True, dev="cpu"):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        if model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, 9216)  # reinitialize the 6th layer
            print(num_ftrs)
            model_ft.to(dev)

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft


# In[ ]:





# In[ ]:




