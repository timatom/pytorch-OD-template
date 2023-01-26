# TODO: Make training script work with inputs and outputs defined by the data and annotations, not by the model itself.
#       That is, the model's inputs and outputs should automatically adjust to the data and annotations.

from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Import the models
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, 
                                          fasterrcnn_mobilenet_v3_large_fpn, 
                                          fasterrcnn_mobilenet_v3_large_320_fpn, 
                                          ssd300_vgg16, 
                                          retinanet_resnet50_fpn, 
                                          maskrcnn_resnet50_fpn, 
                                          keypointrcnn_resnet50_fpn,
                                          ssdlite320_mobilenet_v3_large)

from dataset import Dataset        

batch_size = 4
epoch_num = 10

# Define the transform to be applied to the input data, e.g., ToTensor(), Resize(), etc.
transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])

# Load the COCO dataset
# dataset = CocoDetection(root="/workspaces/pytorch-od-template/datasets/v1/train/", 
#                         annFile="/workspaces/pytorch-od-template/datasets/v1/train/annotations.json", 
#                         transform=transform)
dataset = Dataset(root="/workspaces/pytorch-od-template/datasets/v1/train/",
                  anns_file_path="/workspaces/pytorch-od-template/datasets/v1/train/annotations.json",
                  transform=transform)

# Create a DataLoader for the dataset
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained model
model = ssdlite320_mobilenet_v3_large(weights='COCO_V1', pretrained=True, num_classes=91)

# Replace the classifier with a new one
# num_classes = len(dataset.coco.cats)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
# Move the model to the device
model.to(device)
 
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

def epoch_train(model, data_loader, optimizer, device):
    '''
    Trains the model for one epoch.
    '''
    model.train()
    for images, targets in data_loader:
        # Move the data to the device
        images = list(image.to(device) for image in images)
        
        # Create targets
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        losses = model(images, targets)
         
        # Compute the total loss
        loss = sum(loss for loss in losses.values())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model
        
def train(model, data_loader, optimizer, device, epochs=10):
    '''
    Trains the model for a given number of epochs.
    '''
    print(f"Training the model for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        model = epoch_train(model, data_loader, optimizer, device)
        
    return model

model = train(model, data_loader, optimizer, device, epochs=epoch_num)

# Save the trained model
torch.save(model.state_dict(), 'test_model.pth')
