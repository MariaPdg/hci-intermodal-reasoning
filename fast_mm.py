#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


def fast_mm(matrix_1, matrix_2, device ="cuda:1"):
    
    torch.set_printoptions(precision=15)
    size_1 = matrix_1.size()[0]
    size_2 = matrix_1.size()[1]

    matrix_1.to(device)
    matrix_2.to(device)
    
    if size_1 * size_2 <= 10000 * 100:  #good for 10000*100 matrices
        return torch.matmul(matrix_1, matrix_2.T).reshape(size_1*size_1,1).to(device)
    

    
    
def fast_mm2(matrix_1, matrix_2, device ="cuda:1"): # using repetition (slower)
    
    size_1 = matrix_1.size()[0]
    size_2 = matrix_1.size()[1]
    
    matrix_1.to(device)
    matrix_1 = matrix_1.unsqueeze(1).repeat(1,size_1,1)
        
    matrix_2.to(device)
    
    res = torch.tensor([]).to(device)
    
    for i in range (size_1):
        mult = torch.bmm(matrix_1[i].view(size_1,1,size_2), matrix_2.view(size_1,size_2,1)).to(device)
        res = torch.cat((res, mult),0)
    return res.mean(2)
        
    
