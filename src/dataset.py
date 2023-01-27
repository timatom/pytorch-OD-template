from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, root, anns_file_path, data_transform=None, ann_transform=None):
        '''
        Initializes the dataset object.
        '''
        self.root = root
        self.anns_file_path = anns_file_path
        self.data_transform = data_transform
        self.ann_transform = ann_transform
        
        self.data, self.anns = self.load_data()
        
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns the item at the given index.
        '''
        img = self.data[idx]
        ann = self.anns[idx]

        if self.data_transform:
            img = self.data_transform(img)
        
        if self.ann_transform:
            ann = self.ann_transform(ann)

        return img, ann

    def load_data(self):
        '''
        Loads the data and annotations from the COCO formatted dataset.
        '''
        data = []
        anns = []

        with open(self.anns_file_path, 'r') as f:
            tmp_anns = json.load(f)

        self.categories = tmp_anns["categories"]
        self.images = tmp_anns["images"]

        for image, ann in zip(tmp_anns["images"], tmp_anns["annotations"]):
            if ann is not None:
                img_path = Path(self.root).joinpath(image["file_name"])

                with Image.open(img_path).convert('RGB') as img:
                    data.append(img)
                    
                anns.append(ann)
            
        return data, anns