#!/usr/bin/env python
# coding: utf-8


import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def fast_mm(matrix_1, matrix_2, device ="cuda: 1"):
    size_1 = matrix_1.size()[0]
    size_2 = matrix_1.size()[1]
    
    return torch.matmul(matrix_1, matrix_2.T)
    


def fast_mm2(matrix_1, matrix_2, device ="cuda: 1"):
    size_1 = matrix_1.size()[0]
    size_2 = matrix_1.size()[1]
    
    matrix_1.to(device)
    matrix_1 = matrix_1.unsqueeze(1).repeat(1,size_1,1)
        
    matrix_2.to(device)
    
    res = torch.tensor([]).to(device)
    
    for i in range (size_1):
        mult = torch.bmm(matrix_1[i].view(size_1,1,size_2), matrix_2.view(size_1,size_2,1)).to(device)
        res = torch.cat((res, mult),0)
    return res.mean(1)
        

def vector_idx(i,j, batch_size): #return index in a vector
    return i * batch_size + j 


def matrix_idx(idx,batch_size): #return 2D index in a matrix
    j = idx % batch_size 
    i = idx // batch_size
    return i,j 


def index_by_value(tensor, values):
    return torch.nonzero(tensor == values)[0][0].item()


if __name__ == "__main__":
    error = []
    coeffs = [1, 1, 1, 1, 3, 5, 7, 9, 11, 100, 200, 300]
    for coeff in coeffs:
        matrix_1 = torch.rand(50, 100, dtype=torch.float64)*coeff
        matrix_2 = torch.rand(50, 100, dtype=torch.float64)*coeff
        batch_size = 50
        device = "cpu"

        res1 = fast_mm(matrix_1, matrix_2, device)
        avg_err = []
        for du1 in range(matrix_1.size(0)):
            for du2 in range(matrix_2.size(0)):
                a1 = res1[du1][du2]
                a2 = torch.matmul(matrix_1[du1].view(1, 100), matrix_2[du2].view(100, 1))
                avg_err.append(math.fabs(a1.item() - a2.item()))
        error.append(np.average(avg_err))

    print(error)
    plt.plot(range(len(error)), error)
    plt.savefig("figures/mygraph.png")

