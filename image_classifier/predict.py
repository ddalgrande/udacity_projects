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
import json
import matplotlib.pyplot as plt

import functions_utils

# argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', action='store', dest='img_path', help='path of image to predict', required=True)
parser.add_argument('--cp_dir', action='store', dest='cp_path', help='path of checkpoint', required=True)
parser.add_argument('--top_k', action="store", default=5, dest="top_k", type=int, help='top k predicted values')
parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json', help='add category names')
parser.add_argument('--gpu', action='store_true', default=False, dest='boolean_t', help='Set a switch to use GPU') 
results = parser.parse_args()

img_path = results.img_path
checkpoint_path = results.cp_path
top_k = results.top_k
category_names = results.category_names
gpu = results.boolean_t

# check if GPU is available
if gpu == True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'

# load model check-point
model = functions_utils.load_checkpoint(checkpoint_path)
processed_image = functions_utils.process_image(img_path)
probs, classes = functions_utils.predict(processed_image, model, top_k, device)

# map labels
cat_to_name = functions_utils.labeling(category_names)

labels = []

for class_index in classes:
    labels.append(cat_to_name[str(class_index)])

# convert tensor to numpy-array
print('Name of class: ', labels[0])
print('Probability: ', probs)