# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import json
import math
from PIL import Image
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
#from train import load_checkpoint
#from train import init_checkpoint


parser = argparse.ArgumentParser(description='Predict Image Classifier')

parser.add_argument('--path2image', action='store', default='/content/flowers/test/85/image_04797.jpg')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--saved_models',action='store',default='./checkpoint.pth')

args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


processor_type = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
device = torch.device(processor_type)
saved_models=args.saved_models

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']
model, class_to_idx = load_checkpoint('saved_models')

model.to(device)

#image_path = '/content/flowers/test/85/image_04797.jpg'
image_path = args.path2image
flower_image = Image.open(image_path)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #TODO: Process a PIL image for use in a PyTorch model
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    image = preprocess(image)
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

with Image.open(image_path) as image:
    plt.imshow(image)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    flower_image = Image.open(image_path)
    flower_image = process_image(flower_image)
    flower_image = np.expand_dims(flower_image, 0)
    
    
    flower_image = torch.from_numpy(flower_image)
    
    model.eval()
    inputs = Variable(flower_image).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

topk = args.top_k
probs, classes = predict(image_path, model.to(device),topk)
print(probs)
print(classes)
flower_names = [cat_to_name[str(key)] for key in classes]
print(flower_names)

# TODO: Display an image along with the top 5 classes

def sanity_check(image_path, prob, classes, mapping):
    ''' Function for viewing an image and it's predicted classes.
    '''
    flower_image = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    flower_name = mapping[image_path.split('/')[-2]]
    ax1.set_title(flower_name)
    ax1.imshow(flower_image)
    ax1.axis('off')
    
    y_pos = np.arange(len(prob))
    ax2.barh(y_pos, prob, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()  
    ax2.set_title('Class Probability')

sanity_check(image_path, probs, classes, cat_to_name)

