#!/usr/bin/env python
# coding: utf-8


import torch



def fast_mm(matrix_1, matrix_2, device ="cuda: 1"):
    
    torch.set_printoptions(precision=15)
    size_1 = matrix_1.size()[0]
    size_2 = matrix_1.size()[1]

    matrix_1.to(device)
    matrix_2.to(device)
    
    if size_1 * size_2 <= 10000 * 100:  # good for 10000*100 matrices
        return torch.matmul(matrix_1, matrix_2.T).reshape(size_1*size_1,1).to(device)
    


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



def test1(matrix_1, matrix_2):

    # ### Test #1: matmul

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result_1 = (fast_mm(matrix_1, matrix_2))
    end.record()
    print("Size of matrix A:", matrix_1.size())
    print("Size of matrix B:", matrix_2.size())

    print("Result of fast_mm with matmul:", result_1)
    print("Size of result:", result_1.size())


    torch.cuda.synchronize()
    print("Time:", start.elapsed_time(end))




def test2(matrix_1, matrix_2,):

    # ### Test #2: bmm

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result_2 = fast_mm2(matrix_1,matrix_2)
    end.record()


    print("Size of matrix A:", matrix_1.size())
    print("Size of matrix B:", matrix_2.size())

    print("Result of fast_mm2 with bmm:", result_2)
    print("Size of the result:", result_2.size())


    torch.cuda.synchronize()
    print("Time:", start.elapsed_time(end))


def test3(matrix_1, matrix_2,):

    # ### Test #3: Compare matmul and bmm for small matrices


    res_mm = fast_mm(X,Y)
    print("Result of fast_mm:", res_mm)
    print("Size of the result:", res_mm.size())

    res_bmm = fast_mm2(X,Y)
    print("Result of fast_mm2:",res_bmm)
    print("Size of the result:", res_bmm.size())



def test4(matrix_1, matrix_2, eps = 0.0001):

    # ### Test #4: correctness


    batch_size = matrix_1.size()[0]

    result_1 = (fast_mm(matrix_1, matrix_2)).to(device)
    print("Result of fast_mm:", result_1)
    print("Size of the result:", result_1.size())


    count = 0
    result_1 = (fast_mm(matrix_1, matrix_2)).to(device)
    for i in range(result_1.size()[0]):
        a = matrix_idx(i,batch_size)
        i_1 = a[0]
        i_2 = a[1]
        vect_1 = torch.matmul(matrix_1[i_1],matrix_2[i_2]).to(device)
        #print(vect_1.item())
        if (abs(result_1[i].item() - vect_1.item()) <= eps):
            count = count +1
    print("Number of correspondences with eps:", count)


def test5(matrix_1, matrix_2, num):

    # A[i] * B[j] = C[num] ?

    batch_size = matrix_1.size()[0]

    result_1 = (fast_mm(matrix_1, matrix_2)).to(device)
    print("Vector element", result_1[num].item()) #The element with index num in the vector:"

    pos = matrix_idx(num, batch_size)
    print("Matrix index",pos) #Position of the same element in matrix

    i = pos[0]
    j = pos[1]

    print("Dot product of two vectors",torch.matmul(matrix_1[i],matrix_2[j]).item())



if __name__ == "__main__":


    matrix_1 = torch.rand(500, 100)
    matrix_2 = torch.rand(500, 100)
    batch_size = 500

    X = torch.tensor([[2.0, 3.0], [4.0, 5.0], [7.0, 8.0]])
    Y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    device = "cuda: 1"
    matrix_1.to(device)
    matrix_2.to(device)


    # test1(matrix_1, matrix_2)  # Fast_mm with time
    # test2(matrix_1, matrix_2)  # Fast_mm2 with time (with bmm)
    # test3(X, Y)  # See results of matmul and bmm for small matrices
    test4(matrix_1, matrix_2, eps = 0.0001) # Correctness: number of correspondences for matmul and fast_mm
    # test5(matrix_1, matrix_2, num = 555793) # A[i] * B[j] = C[num]
