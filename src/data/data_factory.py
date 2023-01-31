from pathlib import Path
import json

from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

from src.data import ann_transforms
from src.data.dataset import CustomDataset

class DataFactory:
    def __init__(self, data_path):
        '''
        Initializes the DataFactory object.
        '''
        self.data_path = data_path
        
        self.batch_size = 4
        
        if data_path.joinpath("train").exists():
            self.train_path = data_path.joinpath("train")
        if data_path.joinpath("val").exists():
            self.val_path = data_path.joinpath("val")
        if data_path.joinpath("test").exists():
            self.test_path = data_path.joinpath("test")
            
        # Define the data transforms
        self.data_transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
        
        # Define the annotation transforms
        self.ann_transform = ann_transforms.Compose([ann_transforms.ToTensor, ann_transforms.MatchKeys, ann_transforms.Convert2XYWH])
        
    def get_train_data(self):
        '''
        Returns the training data.
        '''
        dataset = CustomDataset(root=self.train_path,
                  anns_file_path=self.train_path.joinpath("annotations.json"),
                  data_transform=self.data_transform,
                  ann_transform=self.ann_transform)
        
        # Create a DataLoader for the dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
        
        return data_loader
        
    def get_test_data(self):
        '''
        Returns the test data.
        '''
        dataset = CustomDataset(root=self.test_path,
                  anns_file_path=self.test_path.joinpath("annotations.json"),
                  data_transform=self.data_transform,
                  ann_transform=self.ann_transform)
        
        # Create a DataLoader for the dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
        
        return data_loader
        
    def get_validation_data(self):
        '''
        Returns the validation data.
        '''
        dataset = CustomDataset(root=self.val_path,
                  anns_file_path=self.val_path.joinpath("annotations.json"),
                  data_transform=self.data_transform,
                  ann_transform=self.ann_transform)
        
        # Create a DataLoader for the dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
        
        return data_loader
