#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# In[2]:


class Teacher_Net(nn.Module):

    def __init__(self):
        super(Teacher_Net, self).__init__()
        self.linear1 = nn.Linear(in_features=9216, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out = F.relu(self.linear2(out))
        out = F.softmax(self.linear3(out), dim=1)
        return out

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = 1 - nn.functional.cosine_similarity(x_reprets[i], y_reprets[j], dim=-1)

        preds = torch.argmin(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds


# In[3]:


class RankingLossFunc(nn.Module):
    def __init__(self, delta):
        super(RankingLossFunc, self).__init__()
        self.delta = delta

    def forward(self, X, Y):
        assert (X.shape[0] == Y.shape[0])
        loss = 0
        num_of_samples = X.shape[0]
        
        mask = torch.eye(num_of_samples)
        for idx in range(0, num_of_samples):
            negative_sample_ids = [j for j in range(0, num_of_samples) if mask[idx][j] < 1]
            loss += sum([max(0, self.delta
                             - nn.functional.cosine_similarity(X[idx], Y[idx],  dim=-1)
                             + nn.functional.cosine_similarity(X[idx], Y[j], dim=-1)) for j in negative_sample_ids])
        return loss


# In[ ]:

# a copy of the code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class ContrastiveLoss(nn.Module):
    def __init__(self, temp, dev):
        super(ContrastiveLoss, self).__init__()
        self.temp = 0.07
        self.dev = dev
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def predict(self, x_reprets, y_reprets):
        batch_size = x_reprets.shape[0]
        vec_size = x_reprets.shape[1]
        # x_reprets = norm(x_reprets)
        # y_reprets = norm(y_reprets)
        embedding_loss = torch.ones(batch_size, batch_size)
        for i in range(0, batch_size):
            for j in range(0, batch_size):
                embedding_loss[i][j] = nn.functional.cosine_similarity(x_reprets[i], y_reprets[j], dim=-1)
                # print(x_reprets[i], y_reprets[j], torch.matmul(x_reprets[i], y_reprets[j]))
        # print(embedding_loss)
        preds = torch.argmax(embedding_loss, dim=1)  # return the index of minimal of each row
        return preds

    def return_logits(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        return logits, torch.argmax(logits, dim=1)

    def forward(self, q, k, queue):
        N = q.size(0)
        C = q.size(1)
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.T.view(-1, K))
        logits = torch.cat([l_pos.view((N, 1)), l_neg], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.dev)
        loss = self.loss_fn(logits/self.temp, labels)
        # print("loss", loss)
        # print("inside forward", logits.size())
        return loss

    def forward3(self, X, Y, queue):
        assert (X.shape[0] == Y.shape[0] > 0)
        loss = 0
        num_of_samples = X.shape[0]

        for idx in range(num_of_samples):
            pos_logit = torch.matmul(X[idx], Y[idx])
            neg_logits = []
            for count in range(queue.size(0)):
                if count == 1:
                    neg_logits = torch.cat([neg_logits.view(1), torch.matmul(X[idx], queue[count]).view(1)])
                elif count > 1:
                    neg_logits = torch.cat([neg_logits, torch.matmul(X[idx], queue[count]).view(1)])
                else:
                    neg_logits = torch.matmul(X[idx], queue[count])

            logits = torch.cat([pos_logit.view(1), neg_logits])
            loss += F.cross_entropy(logits.view((1, logits.size(0)))/self.temp,
                                    torch.zeros(1, dtype=torch.long, device=self.dev))

        return loss/num_of_samples



# class CustomedQueue:
#     def __init__(self):
#         self.neg_keys = []
#         self.size = 0

#     def empty(self):
#         return self.size == 0

#     def enqueue(self, new_tensor):
#         if self.size == 0:
#             self.neg_keys = new_tensor
#         else:
#             self.neg_keys = torch.cat([self.neg_keys, new_tensor])
#         self.size += new_tensor.size(0)

#     def dequeue(self, howmany=1):
#         if self.size > 0:
#             self.size -= howmany
#             self.neg_keys = self.neg_keys[howmany:]

#     def get_tensor(self):
#         return torch.transpose(self.neg_keys, 0, 1)

class CustomedQueue:
    def __init__(self, max_size=1024):
        self.neg_keys = []
        self.size = 0
        self.max_size = max_size

    def empty(self):
        return self.size == 0

    def enqueue(self, new_tensor):
        if self.size == 0:
            self.neg_keys = new_tensor
        else:
            self.neg_keys = torch.cat([self.neg_keys, new_tensor])
        self.size = self.neg_keys.size(0)

    def dequeue(self, howmany=1):
        if self.size > self.max_size:
            self.neg_keys = self.neg_keys[-self.max_size:]
            self.size = self.neg_keys.size(0)
            # print("m",self.neg_keys[-howmany:])
            # print("p",self.neg_keys[howmany:])

    def get_tensor(self, transpose=False):
        if transpose:
            return torch.transpose(self.neg_keys, 0, 1)
        else:
            return self.neg_keys
