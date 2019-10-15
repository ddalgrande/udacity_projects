# coding: utf-8

# import packages
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import argparse
import matplotlib.pyplot as plt

import functions_utils

# argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', action='store', dest='path', help='path of directory', required=True)
parser.add_argument('--save_dir', action='store', dest='cp_path', default='./checkpoint/', help='path of checkpoint')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16', choices={"vgg16", "densenet161", "alexnet"}, help='architecture of the network')
parser.add_argument('--learning_rate', action='store', nargs='?', default=0.001, type=float, dest='learning_rate', help='(float) learning rate of the network')
parser.add_argument('--epochs', action='store', dest='epochs', default=3, type=int, help='(int) number of epochs while training')
parser.add_argument('--hidden_units', action='store', nargs=2, default=[10240, 1024], dest='hidden_units', type=int,
                    help='Enter 2 hidden units of the network, input -> --hidden_units 256 256 | output-> [512, 256]')
parser.add_argument('--gpu', action='store_true', default=True, dest='boolean_t', help='Set a switch to use GPU')
results = parser.parse_args()


data_dir = results.path
checkpoint_path = results.cp_path
arch = results.arch
hidden_units = results.hidden_units
epochs = results.epochs
lr = results.learning_rate
gpu = results.boolean_t
print_every = 30

# check if GPU is available
if gpu == True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');

else:
    print('GPU Off');
    device = 'cpu'
        
# load dataset
data_train_loader, data_test_loader, data_validation_loader, data_train_set = functions_utils.loading_data(data_dir)
class_to_idx = data_train_set.class_to_idx

# setup network
model, input_size, criterion, optimizer = functions_utils.model_setup(arch, hidden_units, lr)

# train model
functions_utils.train_model(model, data_train_loader, data_validation_loader, epochs, print_every, criterion, optimizer, device)

# test model
functions_utils.validate_model(model, data_test_loader)

# save check-point
functions_utils.save_checkpoint(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path)