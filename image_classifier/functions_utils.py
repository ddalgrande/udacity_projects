# coding: utf-8

# import modules
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

using_gpu = torch.cuda.is_available()


def loading_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # define transforms for  training, validation and test sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    # load datasets with ImageFolder
    data_train_set = datasets.ImageFolder(train_dir, transform = train_transforms)
    data_validation_set = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    data_test_set = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    # define the dataloaders using datasets and transforms 
    data_train_loader = torch.utils.data.DataLoader(data_train_set, batch_size = 64, shuffle = True)
    data_test_loader = torch.utils.data.DataLoader(data_test_set, batch_size = 64, shuffle = True)
    data_validation_loader = torch.utils.data.DataLoader(data_validation_set, batch_size = 64, shuffle=True)
    
    return data_train_loader, data_test_loader, data_validation_loader, data_train_set


# build and train network - freeze parameters so it doesn't backpropagate through them
def model_setup(arch, hidden_units, lr):
    
    if arch == "vgg16":
        model = models.vgg16(pretrained = True)
        input_size = 25088
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 2208
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_size = 2208
    else:
        print("{} is not a valid model. Did you mean vgg16, densenet121 or alexnet?".format(structure))
        
    output_size = 102

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('dropout',nn.Dropout(0.5)),
                                            ('fc1', nn.Linear(input_size, hidden_units[0])),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                            ('relu2', nn.ReLU()),
                                            ('fc3', nn.Linear(hidden_units[1], output_size)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    
    return model, input_size, criterion, optimizer


# train model
def train_model(model, data_train_loader, data_validation_loader, epochs, print_every, criterion, optimizer, device):
    
    epochs = epochs
    print_every = print_every
    steps = 0
    loss_show = []
    
    # switch to cuda
    model = model.to('cuda')
    
    for e in range(epochs):
        
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(data_train_loader):
            
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                
                model.eval()
                validation_loss = 0
                accuracy = 0

                for i, (inputs,labels) in enumerate(data_validation_loader):

                            optimizer.zero_grad()
                            inputs, labels = inputs.to('cuda') , labels.to('cuda')
                            model.to('cuda')

                            with torch.no_grad():
                                
                                # validate
                                outputs = model.forward(inputs)
                                validation_loss = criterion(outputs,labels)
                                ps = torch.exp(outputs).data
                                equality = (labels.data == ps.max(1)[1])
                                accuracy += equality.type_as(torch.FloatTensor()).mean()

                # calculate validation loss
                val_loss = validation_loss / len(data_validation_loader)
                train_ac = accuracy /len(data_validation_loader)

                # print number of epochs, training loss, validation loss and accuracy
                print("Epoch: {}/{}... | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss {:.4f} | ".format(val_loss),
                      "Accuracy {:.4f}".format(train_ac))

                running_loss = 0


# perform validation on the test set
def validate_model(model, dataloader):

    model.eval()
    model.to('cuda')
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _ , prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.data).sum().item()

        # print accuracy on test set
        print('Test Set Accuracy: %d %%' % (100 * correct / total))   


# save checkpoint 
def save_checkpoint(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path):
    
    model.class_to_idx = class_to_idx

    state = {'structure' : arch,
            'learning_rate': lr,
            'epochs': epochs,
            'input_size': input_size,
            'hidden_units': hidden_units,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx
            }

    torch.save(state, 'command_checkpoint.pth')

    print('Check-point saved with name command_checkpoint.pth')

    
# load checkpoint and re-build model
def load_checkpoint(path):
    
    # load parameters
    state = torch.load(path)
    lr = state['learning_rate']
    input_size = state['input_size']
    structure = state['structure']
    hidden_units = state['hidden_units']
    epochs = state['epochs']
    
    # build model from check-points
    model,_,_,_ = model_setup(structure, hidden_units, lr)
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])

    return model


# process image before running PyTorch model
def process_image(image):

    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array.
    '''
    
    # process a PIL image to use it in a PyTorch model
    pil_image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                                std = [0.229, 0.224, 0.225])])
    img = image_transforms(pil_image)

    return img


# labelling
def labeling(category_names):

    with open(category_names, 'r') as f:

        cat_to_name = json.load(f)

        return cat_to_name


# predict class
def predict(processed_image, model, topk, device):

    ''' 
    Predict the class (or classes) of an image using a pre-trained deep learning model.
    '''
    
    # implement  code to predict class from an image
    model.eval()
    model.cpu()
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():

        output = model.forward(processed_image)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return top_prob, top_classes