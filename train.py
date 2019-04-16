# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from utils import query_yes_no
from collections import OrderedDict
import os

parser = argparse.ArgumentParser(description='Train Image Classifier')

parser.add_argument('data_directory')
parser.add_argument('--save_dir', action='store', default='.')
parser.add_argument('--arch', action='store', default='vgg19')
parser.add_argument('--learning_rate', action='store', default=0.01, type=float)
parser.add_argument('--hidden_units', action='append', default=[], type=int)
parser.add_argument('--epochs', action='store', default=10, type=int)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

incorrect_data_dir = not os.path.exists(train_dir) \
                   or not os.path.exists(valid_dir) \
                   or not os.path.exists(test_dir)

try:
    if incorrect_data_dir:
        print('Eww Error: This is incorrect data directory. ' + args.data_directory)
        print('Data directory should have train, valid and test sub directories')
        exit(1)
except OSError:
    print('Error: Checking data directory.')
    exit(1)

data_transforms = {
    "training": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    "validation": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train','valid','test']}

batch_size = 64
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train','valid','test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train','valid','test']}
class_names = image_datasets['train'].classes
device = torch.device("cuda")

if len(args.hidden_units) == 0:
    args.hidden_units = [512]


#def pretrained_model(name):
#    models_choice = {'vgg19', 'densenet161'}
#    if name not in models_choice:
#        print("Not trained model: Can only choose 'vgg19 or 'densenet121'")
#        exit(1)
#    return getattr(models, name)

if args.arch not in('vgg' or 'densenet'):
    parser.error("Not trained model: Can only choose 'vgg19 or 'densenet121'")


init_lr = args.learning_rate
epochs = args.epochs
if args.epochs > 20:
    parser.error("Please choose value less than 20")

model = models(args.arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

if 'vgg' in args.arch:
    input_features = 25088
elif 'densenet' in args.arch:
    input_features = 2208

from torch.optim import lr_scheduler


hidden_layers = []
previous_layer = input_features

for i, layer in enumerate(args.hidden_units):
    hidden_layers.append(('class_layer_' + str(i + 1), nn.Linear(previous_layer, hidden_layers)))
    hidden_layers.append(('relu_' + str(i + 1), nn.ReLU()))
    previous_layer = layer

hidden_layers.append(('fc_last', nn.Linear(previous_layer, 102)))
hidden_layers.append(('output', nn.LogSoftmax(dim=1)))

classifier = nn.Sequential(OrderedDict(hidden_layers))
model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=init_lr)
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

check_gpu = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'


if args.gpu and check_gpu == 'cpu':
    confirm_cpu = query_yes_no('GPU doesnot exists, do you wish to continue using CPU?')
    if not confirm_cpu:
        exit(0)

device = torch.device(check_gpu)
model.to(device)



#images.cuda()
# Referrence https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, sched, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid','test']:
            if phase == 'train':
                model.train()  
            else:
              model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                                       
              

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

print('No. of Epochs: ', args.epochs)
print('Learning rate: ', args.learning_rate)
print('Pre-trained model: ', args.arch)
print('Classification layers: ', args.hidden_units)
print('/n' + 'Start of the Image Classification model' + '/n')

epochs = 10
model.to(device)
model = train_model(model, criterion, optimizer, sched, epochs)

# TODO: Do validation on the test set

model.eval()

accuracy = 0

for inputs, labels in dataloaders['test']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    
    # Class with the highest probability is our predicted class
    equality = (labels.data == outputs.max(1)[1])

    # Accuracy is number of correct predictions divided by all predictions
    accuracy += equality.type_as(torch.FloatTensor()).mean()
    
print("Accuracy of the model on Test Data: {:.3f}".format(accuracy/len(dataloaders['test'])))


confirm_save = query_yes_no("Should we now save the model!")

if not confirm_save:
    exit(0)

model.class_to_idx = image_datasets['training'].class_to_idx


def init_checkpoint():
    checkpoint = {
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'batch_size': 32,
        'output_size': 102,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'arch': args.arch,
        'scheduler': sched,
        'model': models(args.arch)(pretrained=True),
        'classifier': classifier,
    }

    torch.save(checkpoint, checkpoint.pth)

init_checkpoint()
checkpoint = torch.load('checkpoint.pth')
checkpoint.keys()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, checkpoint['class_to_idx']

model, class_to_idx = load_checkpoint('checkpoint.pth')
model