#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random

from tqdm import tqdm_notebook as tqdm
import torch
import torchtext
from torchtext.vocab import GloVe
from transformers import DistilBertTokenizer
from tqdm import tqdm, tqdm_notebook
import numpy as np


def build_vocab(train_cap):
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    vocab = []
    for i in range (len(train_cap)):
        words = tokenizer.decode(train_cap[i].numpy()).split(' ')
        for j in range (len(words)):
            if  words[j]== '[CLS]' or  words[j]== '[PAD]' or words[j]=='[SEP]':
                continue;
            else:  
                if words[j] not in vocab:  #add unique words
                    vocab.append(words[j])

    vocab.append('[CLS]')
    vocab.append('[PAD]')
    vocab.append('[SEP]')

    return vocab


def build_embed(vocab):
    
    glove = torchtext.vocab.Vectors('../../words/glove/glove.6B.300d.txt')
#     print(len(glove[vocab[10]]))

    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    i = 0
    for word in vocab:
        embedding_vector = glove[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            i = i+1
    return embedding_matrix

#embedding_matrix = build_embed(vocab)
#print(np.shape(embedding_matrix))


def build_train(train_cap):
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    vocab = []
    for i in range (len(train_cap)):
        words = tokenizer.decode(train_cap[i].numpy()).split(' ')
        vocab.append(words)
    return vocab

# train = build_train(train_cap)
# print(len(train))


def replace_values(train_cap,vocab):

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    for i in range (train_cap.size()[0]):
        words = tokenizer.decode(train_cap[i].numpy()).split(' ')
        for j in range(len(words)):
            if words[j] in vocab:
                train_cap[i][j] = vocab.index(words[j])
        # add padding 
        if len(words)<len(train_cap[i]): 
            for j in range(len(words),len(train_cap[i])):
                train_cap[i][j] = vocab.index('[PAD]')
                
    return train_cap
    