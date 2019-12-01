
#imports
import torch
from PIL import Image
from torchvision import datasets, transforms, models
import json
import argparse

print("Image Classification Predicter")

parser = argparse.ArgumentParser()
parser.add_argument("path_to_image", 
                    help="path to image to run prediction on", 
                    type=str)
parser.add_argument("checkpoint", 
                    help="path to saved training checkpoint of model to use", 
                    type=str)
parser.add_argument("--category_names",
                    help="path of json file containing categories to real names translation (default: cat_to_name.json)",
                    default="./cat_to_name.json",
                    type=str)
parser.add_argument("--top_k",
                    help="number of top K classes to return (default: 5)", 
                    default=5,
                    type=int)

parser.add_argument("--gpu",  
                    help="use gpu for prediction \n(default:False)",
                    default=False, 
                    action='store_true')

args = parser.parse_args()
#todo this should be a parameter too
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    """
    loads a checkpoint and rebuilds the model
    """
    
    # load the saved checkpoint
    model_checkpoint = torch.load(path)
    
    arch =  model_checkpoint['architecture']
       
    # load pretrained model
    model = models.__dict__[arch](pretrained=True)
    
    # freeze parameters, no backpropagation
    for param in model.parameters(): 
        param.requires_grad = False
    
    # load state variables from checkpoint
    model.class_to_idx = model_checkpoint['class_to_idx']
    model.classifier = model_checkpoint['classifier']
    model.load_state_dict(model_checkpoint['state_dict'])
    
    if args.gpu:
        model.cuda()
    else:
        model.cpu()

    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    (n_means, n_stddev) = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
           
    source_image = Image.open(image).convert("RGB")
        
    image_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(n_means, n_stddev)])
    
    source_image = image_transforms(source_image)
    
    if args.gpu:
        source_image= source_image.cuda()
        
    return source_image

def predict(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: path to image.
    model: neural network model
    top_k: The top K classes to be calculated
    '''
   
       
    # set model to evaluation mode
    model.eval();

    # load image as torch.Tensor
    image = process_image(image_path)
    
    # return a new tensor with a single size one dimension
    image = image.unsqueeze(0)
    # disable gradient calculation 
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, top_k)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
        
    labels = top_labels.cpu().numpy()[0]
    probabilities =  top_prob.cpu().numpy()[0]
    
    #create a mapped classes dictionary
    class_to_idx_dict = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    output_classes = []
    
    for label in labels:
        output_classes.append(class_to_idx_dict[label])
        
    return probabilities, output_classes

print("loading checkpoint from: " + args.checkpoint)

model = load_checkpoint(args.checkpoint)

image_path = args.path_to_image 


# make prediction
probabilities, classes = predict(image_path, model, args.top_k )
label = classes[0]

labels_and_probabilities = []

for iterator, class_idx in enumerate(classes):
    labels_and_probabilities.append("{:20}: {:.5f}\n".format(cat_to_name[class_idx], probabilities[iterator]))

print(("\nPrediction\n_______________________\n"+
"Flower name: {} \n" +
"Probability: {:.5f}\n\nTop {} predictions\n_______________________").format(cat_to_name[label],probabilities[0], args.top_k))

print (''.join(labels_and_probabilities))


