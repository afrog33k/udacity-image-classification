#imports
import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import json
import argparse
import os


print("Image Classification Trainer")
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", 
                    help="path to directory of inputs in test, train and valid sub-folders\n(default: flowers)", 
                    default="flowers",
                    type=str)
parser.add_argument("--save_dir",
                    help="set directory to save checkpoints\n(default: ./saved_model)",
                    default="./saved_model",
                    type=str)
parser.add_argument("--arch",
                    help="choose architecture e.g. vgg16, vgg13, densenet, mobilenet etc...\n(default: vgg16)", 
                    default="vgg16",
                    type=str)
parser.add_argument("--learning_rate",
                    help="set learning rate \n(default:0.001)",
                    default="0.001", 
                    type=float)
parser.add_argument("--hidden_units",
                    help="set number of hidden units \n(default:4096)",
                    default="4096",
                    type=int)
parser.add_argument("--epochs",
                    help="set number of epochs to use \n(default:5)",
                    default=5,
                    type=int)
parser.add_argument("--gpu",  
                    help="use gpu for training \n(default:False)",
                    default=False, 
                    action='store_true')

args = parser.parse_args()
# todo turn this into a parameter
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print("define transforms for the training data and testing data")
# define transforms for the training data and testing data
normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalization])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalization])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalization])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

#Larger batch sizes seem to get accurate much faster
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

print("loading label to name translation")
#todo this should be a parameter too
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
pretrained_network = args.arch

print("loading a pre-trained network: " + pretrained_network)

# load a pre-trained network (transfer learning)
model = models.__dict__[pretrained_network](pretrained=True)
model.name = args.arch

# freeze parameters to stop backpropagation
for param in model.parameters():
    param.requires_grad = False

number_of_classes = int(max(cat_to_name, key=int))

output_size = number_of_classes

#get input size from current classifier
input_size = model.classifier[0].in_features

#get dropout probability from current classifier
dropout_p = model.classifier[2].p

#get size of hidden layers
hidden_size = args.hidden_units #model.classifier[0].out_features

# define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
                          ('linear1', nn.Linear(input_size, hidden_size, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout', nn.Dropout(p=dropout_p)),
                          ('linear2', nn.Linear( hidden_size, output_size, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

# detect and switch model to cuda if available (not compatible with command line)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Force device if --gpu is set
if args.gpu:
    device = "cuda:0"
else:
    device = "cpu"

# change to device type
model.to(device);


print("loading neural network device: " + device)


# create adam optimizer and select loss function
criterion = nn.NLLLoss()
#todo: learning rate needs to be a parameter
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# deep learning parameters
#todo: these need to be parameters as well
epochs = args.epochs
print_step = 64 # prints status every 64 images


# validation function
def validation(model, validation_loader, criterion):
    validation_loss,accuracy = 0,0 
    
    for iterator, (inputs, labels) in enumerate(validation_loader):
        
        #convert inputs and labels to appropriate device
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        
        validation_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy


# Train the model using backpropogation
def train_model():
    print("started training model")
    for epoch in range(epochs):
        accumulated_loss = 0
        model.train() 
    
        for iterator, (inputs, labels) in enumerate(train_loader):
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
        
            loss = criterion(outputs, labels)
        
            loss.backward()
        
            optimizer.step()
        
            accumulated_loss += loss.item()
        
            if iterator % print_step == 0:
                model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validation(model, valid_loader, criterion)
            
                print("epoch: {}/{}, training loss: {:.2f}, validation loss: {:.2f}, validation accuracy: {:.2f}".format(epoch+1, epochs, 
                                                                    accumulated_loss/print_step,
                                                                    validation_loss/len(valid_loader),
                                                                    accuracy/len(valid_loader)))
                accumulated_loss = 0
                model.train()

    print("\ntraining model complete\n")
    
def validate_model():
    print("validating model")
    # TODO: Do validation on the test set
    correct_images, total_images = 0 , 0
    # disable gradient calculation
    with torch.no_grad():
        model.eval()
        for iterator, (images, labels) in enumerate(test_loader):
        
            #convert to correct device
            images, labels = images.to(device), labels.to(device)
        
            outputs = model(images)
        
            _, predicted = torch.max(outputs.data, 1)
        
            total_images += labels.size(0)
        
            correct_images += (predicted == labels).sum().item()

    print('Accuracy of {:.2f}% on {} images.'.format((100 * correct_images / total_images), total_images))
    
    
#save the model checkpoint
def save_model():
    print("saving model")
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'architecture': model.name,
        'classifier': model.classifier,    
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()}
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = args.save_dir + '/model_checkpoint.pth'
    torch.save(checkpoint, args.save_dir + '/model_checkpoint.pth')
    
    print("saved checkpoint to {}".format(save_path))

train_model()
validate_model()
save_model()
