'''
Use simple pytorch-based API?:
* skorch
* pytorch lightning, lightning flash
* fastai
* ignite

[] 1D dense autoencoder
[] 1D convolutional autoencoder

[] FastText
[] Sparse AutoEncoder
[] Variational AutoEncoder
[] 1D CNN
[] 2D CNN
[] WaveNet
[] TabNet
[] TCN
'''

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

# class dense_1D(BaseEstimator, TransformerMixin, nn.Module):
#     def __init__(self):
#         return True
#     def fit(self, X, y=0):
#     def fit_transform(self, X, y=0):
        

