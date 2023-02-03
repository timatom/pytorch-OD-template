from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# TODO: Make training script work with inputs and outputs defined by the data and annotations, not by the model itself.
#       That is, the model's inputs and outputs should automatically adjust to the data and annotations.
class ModelFactory:
    def __init__(self, data_factory, model, loss_fn):
        '''
        Initializes the ModelFactory.
        '''
        
        self.data_factory = data_factory
        
        # Define the device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = model
        
        # Move the model to the device
        self.model.to(self.device)
        
        self.loss_fn = loss_fn
    
    def epoch_train(self, data_loader, optimizer, device):
        '''
        Trains the model for one epoch.
        '''
        self.model.train()
        for images, targets in data_loader:
            # Move the data to the device
            images = list(image.to(device) for image in images)
            
            # Create targets
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute the loss
            losses = self.loss_fn(outputs, targets)
            
            # Compute the total loss
            loss = sum(loss for loss in losses.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def train(self, epochs=10, lr=0.005, momentum=0.9, weight_decay=0.0005):
        '''
        Trains the model for a given number of epochs.
        '''
        train_data = self.data_factory.get_train_data()
        
        # Define the optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        print(f"Training the model for {epochs} epochs...")
        for _ in tqdm(range(epochs)):
            self.epoch_train(train_data, optimizer, self.device)
    
    def validate(self):
        # TODO: Implement validation
        '''
        Validates the model.
        '''
        return None
    
    def test(self):
        # TODO: Implement testing
        '''
        Tests the model.
        '''
        return None
            
    def save_model(self, path):
        '''
        Saves the model.
        '''
        torch.save(self.model, path)
        
    def load_model(self, path):
        '''
        Loads the model.
        '''
        self.model = torch.load(path)
    
    def deploy(self):
        # TODO: Implement deployment
        '''
        Deploys the model.
        '''
        return None
