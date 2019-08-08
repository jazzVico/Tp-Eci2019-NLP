import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fasttext
import io
import argparse 


torch.manual_seed(42)

# def load_vectors(fname):
#     i=0
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#         i+=1
#         if i==100:      
#             return data

# vectors = load_vectors("wiki-news-300d-1M.vec")


model = fasttext.train_supervised('pepe.txt')